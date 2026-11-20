"""Build weak Alive/Dead crown crops from NEON VST stems across every RGB flight year.

For each candidate stem (see ``fetch_vst_tiles.py``) and each RGB year Y, the
label comes from the field visits around Y (``year_labels``):

- Dead: the last visit on or before Y is standing dead or dead with a broken bole,
  and no later visit is Live.
- Alive: the last visit on or before Y is Live and the next visit is Live too, or
  there is no next visit. This gives paired live crops of the same crowns from
  before they died.
- Years more than ``--max-years-after`` past the last visit are skipped, since the
  tree may have fallen or changed. If the next visit finds the tree downed or lost,
  only flights within a year of the last standing-dead visit are kept.

Crowns are found with the DeepForest tree detector in a window around each stem.
Stems and boxes are then matched one-to-one (Hungarian assignment), allowing at
most ``--max-dist-m`` between stem and box centre. The CHM maximum in the box is
compared with the VST height. Crowns are flagged (``flag_*``) when the match is
too far, the height disagrees, or the box is small, a sliver, or cut by the
window edge.

Crops on 1 km tiles used for validation or test in ``--holdout-split`` are dropped.
Optionally each crop is scored with an existing CropModel (``p_dead``) for weak
mining.

Writes ``crops/neon_vst/{Alive,Dead}/`` and ``manifests/neon_vst[_unfiltered].csv``.

Usage:
    python build_vst_crops.py --root /blue/ewhite/b.weinstein/DeepForest/alive_dead \
        --score-model weecology/cropmodel-deadtrees
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from build_treefinder_crops import detect, int_boxes
from PIL import Image
from rasterio.windows import Window
from scipy.optimize import linear_sum_assignment
from torchvision.ops import nms

from deepforest import main as df_main

DEAD = ["Standing dead", "Dead, broken bole"]
LIVE = ["Live"]


def year_labels(
    merged: pd.DataFrame, stems: pd.DataFrame, flights: pd.DataFrame, max_years_after: int
) -> pd.DataFrame:
    """One row per (stem, tile, RGB year) with a weak label from the bracketing visits."""
    v = merged[merged.individualID.isin(stems.individualID)].copy()
    v["vyear"] = pd.to_datetime(v.date, utc=True, format="mixed").dt.year.astype("int64")
    # One status per individual-year: a dead record that year wins over live.
    v["rank"] = v.plantStatus.isin(DEAD).astype(int)
    v = v.sort_values(["vyear", "rank"]).drop_duplicates(
        ["individualID", "vyear"], keep="last"
    )
    v = v[["individualID", "vyear", "plantStatus", "height"]].sort_values("vyear")

    q = stems.merge(flights, on=["domainID", "tileE", "tileN"])
    q = q.sort_values("year").reset_index(drop=True)
    q = pd.merge_asof(
        q,
        v.rename(columns={"vyear": "prev_year"}),
        left_on="year",
        right_on="prev_year",
        by="individualID",
        direction="backward",
    )
    nxt = pd.merge_asof(
        q[["individualID", "year"]],
        v.rename(
            columns={
                "vyear": "next_year",
                "plantStatus": "next_status",
                "height": "next_height",
            }
        ),
        left_on="year",
        right_on="next_year",
        by="individualID",
        direction="forward",
        allow_exact_matches=False,
    )
    q[["next_year", "next_status"]] = nxt[["next_year", "next_status"]].to_numpy()
    q["years_since_visit"] = q.year - q.prev_year
    later_live = q.next_status.isin(LIVE) | q.next_status.str.startswith("Live", na=False)
    # If it is next seen downed or lost, it may already have fallen: trust only nearby flights.
    fell = q.next_status.notna() & ~q.next_status.isin(DEAD) & ~later_live
    dead = q.plantStatus.isin(DEAD) & ~later_live & (~fell | (q.year - q.prev_year <= 1))
    alive = q.plantStatus.isin(LIVE) & (q.next_status.isna() | q.next_status.isin(LIVE))
    q["label"] = np.select([dead, alive], ["Dead", "Alive"], "")
    q = q[(q.label != "") & (q.years_since_visit <= max_years_after)]
    return q.rename(columns={"plantStatus": "vst_status", "height": "vst_height"})


def read_rgb(src, window: Window, **kw) -> np.ndarray:
    """HxWx3 uint8 window; some NEON tiles store the same bytes as int8."""
    a = src.read([1, 2, 3], window=window, **kw)
    return np.moveaxis(a.view(np.uint8) if a.dtype == np.int8 else a, 0, -1)


def chm_in_boxes(chm_path: str | None, boxes_world: np.ndarray) -> np.ndarray:
    """Maximum CHM height inside each box given in map coordinates (xmin, ymin, xmax, ymax)."""
    out = np.full(len(boxes_world), np.nan)
    if not isinstance(chm_path, str):
        return out
    with rasterio.open(chm_path) as src:
        for i, (x0, y0, x1, y1) in enumerate(boxes_world):
            win = (
                rasterio.windows.from_bounds(x0, y0, x1, y1, src.transform)
                .round_offsets()
                .round_lengths()
            )
            a = src.read(1, window=win, boundless=True, fill_value=src.nodata or -9999)
            a = a[a > -1]
            out[i] = a.max() if a.size else np.nan
    return out


def tile_year_crops(
    rgb_path: str, chm_path: str | None, rows: pd.DataFrame, model, args, device
) -> list[dict]:
    """Detect, match and crop every labelled stem on one RGB tile-year."""
    with rasterio.open(rgb_path) as src:
        gsd = src.res[0]
        t = src.transform
        col, row = ~t * (rows.stemEasting.to_numpy(), rows.stemNorthing.to_numpy())
        col, row = np.asarray(col), np.asarray(row)
        inside = (col >= 0) & (row >= 0) & (col < src.width) & (row < src.height)
        rows, col, row = rows[inside], col[inside], row[inside]
        if rows.empty:
            return []
        half = int(args.window_m / gsd / 2)
        boxes = []
        # Stems in the same plot share windows: detect once per window position.
        keys = np.column_stack([col // half, row // half]).astype(int)
        for key in np.unique(keys, axis=0):
            c0, r0 = (key * half - half // 2).clip(0)
            w = Window(
                c0, r0, min(2 * half, src.width - c0), min(2 * half, src.height - r0)
            )
            rgb = read_rgb(src, w)
            if rgb.max() == 0:
                continue
            img = (
                torch.from_numpy(np.ascontiguousarray(rgb))
                .permute(2, 0, 1)
                .float()
                .to(device)
                / 255
            )
            det = detect(model, img)
            det = det[det[:, 4] >= args.min_score]
            det[:, [0, 2]] += c0
            det[:, [1, 3]] += r0
            # Boxes touching an inner window edge are partial; full ones come from a neighbour window.
            ok = (
                ((det[:, 0] > c0 + 1) | (c0 == 0))
                & ((det[:, 1] > r0 + 1) | (r0 == 0))
                & ((det[:, 2] < c0 + w.width - 1) | (c0 + w.width == src.width))
                & ((det[:, 3] < r0 + w.height - 1) | (r0 + w.height == src.height))
            )
            boxes.append(det[ok])
        if not boxes:
            return []
        b = np.concatenate(boxes)
        keep = nms(torch.from_numpy(b[:, :4]), torch.from_numpy(b[:, 4]), 0.3).numpy()
        b = b[keep]

        # One-to-one stem-to-crown assignment by centre distance.
        cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
        d = np.hypot(col[:, None] - cx[None], row[:, None] - cy[None]) * gsd
        inside_box = (
            (col[:, None] >= b[None, :, 0])
            & (col[:, None] <= b[None, :, 2])
            & (row[:, None] >= b[None, :, 1])
            & (row[:, None] <= b[None, :, 3])
        )
        cost = np.where(d <= args.max_dist_m, d - inside_box * 0.5, 1e6)
        ri, bi = linear_sum_assignment(cost)
        ok = cost[ri, bi] < 1e6
        ri, bi = ri[ok], bi[ok]
        if not len(ri):
            return []
        mb = b[bi]
        x0, y0 = t * (mb[:, 0], mb[:, 1])
        x1, y1 = t * (mb[:, 2], mb[:, 3])
        chm = chm_in_boxes(
            chm_path, np.column_stack([x0, np.minimum(y0, y1), x1, np.maximum(y0, y1)])
        )
        padded = int_boxes(mb[:, :4], args.pad, src.width, src.height)
        out = []
        for k, (i, (px0, py0, px1, py1)) in enumerate(zip(ri, padded, strict=True)):
            crop = read_rgb(src, Window(px0, py0, px1 - px0, py1 - py0))
            r = rows.iloc[i]
            out.append(
                {
                    "crop": crop,
                    "individualID": r.individualID,
                    "label": r.label,
                    "year": int(r.year),
                    "site": r.siteID,
                    "flight_site": r.site,
                    "group": f"{r.siteID}_{int(r.tileE)}_{int(r.tileN)}",
                    "easting": r.stemEasting,
                    "northing": r.stemNorthing,
                    "xmin": int(px0),
                    "ymin": int(py0),
                    "xmax": int(px1),
                    "ymax": int(py1),
                    "crown_xmin": float(mb[k, 0]),
                    "crown_ymin": float(mb[k, 1]),
                    "crown_xmax": float(mb[k, 2]),
                    "crown_ymax": float(mb[k, 3]),
                    "det_score": float(mb[k, 4]),
                    "match_dist_m": float(d[i, bi[k]]),
                    "stem_in_box": bool(inside_box[i, bi[k]]),
                    "chm_max": float(chm[k]),
                    "vst_height": r.vst_height,
                    "vst_status": r.vst_status,
                    "prev_year": int(r.prev_year),
                    "next_status": r.next_status,
                    "years_since_visit": int(r.years_since_visit),
                    "rgb_path": rgb_path,
                    "gsd_m": gsd,
                }
            )
        return out


def flag(df: pd.DataFrame, args) -> pd.DataFrame:
    """Add ``flag_*`` columns for unreliable matches and partial crowns."""
    w = df.crown_xmax - df.crown_xmin
    h = df.crown_ymax - df.crown_ymin
    resid = df.chm_max - df.vst_height
    tol = np.maximum(args.height_tol_m, args.height_tol_frac * df.vst_height)
    # Broken boles and decaying snags read lower in the CHM than the recorded height.
    return df.assign(
        height_resid=resid,
        flag_height=(resid.abs() > tol) & df.chm_max.notna() & df.vst_height.notna(),
        flag_small=np.sqrt(w * h) * df.gsd_m < args.min_side_m,
        flag_aspect=np.maximum(w, h) / np.minimum(w, h).clip(1) > args.max_aspect,
    )


@torch.no_grad()
def score(df: pd.DataFrame, root: Path, model_name: str) -> np.ndarray:
    """P(Dead) for each crop from an existing CropModel."""
    from evaluate_cropmodel import load, predict

    return predict(load(model_name), [str(root / p) for p in df.crop_path])


def main(args: argparse.Namespace) -> None:
    src = args.root / "sources"
    vst_dir = src / "neon_vst" / args.release
    aop = src / "neon_aop"
    stems = pd.read_csv(aop / "stem_tiles.csv")
    man = pd.read_csv(aop / "tile_manifest.csv")
    man = man[man.status.isin(["local", "downloaded"])]
    rgb = man[man["product"] == "rgb"][
        ["domainID", "site", "year", "tileE", "tileN", "dest"]
    ]
    chm = man[man["product"] == "chm"][
        ["domainID", "year", "tileE", "tileN", "dest"]
    ].rename(columns={"dest": "chm_path"})
    flights = rgb.merge(chm, on=["domainID", "year", "tileE", "tileN"], how="left")

    # Each stem is processed only on the tile that contains it.
    stems = stems[
        (stems.stemEasting // 1000 * 1000 == stems.tileE)
        & (stems.stemNorthing // 1000 * 1000 == stems.tileN)
    ]
    merged = pd.read_csv(
        vst_dir / "vst_merged.csv",
        low_memory=False,
        usecols=["individualID", "date", "plantStatus", "height"],
    )
    q = year_labels(merged, stems, flights, args.max_years_after)

    holdout = pd.read_csv(args.root / "splits" / args.holdout_split / "split.csv")
    held = set(
        holdout.loc[holdout.split.isin(["val", "test"]), "group"].str.split(":").str[-1]
    )
    q = q[
        ~(
            q.siteID
            + "_"
            + q.tileE.astype(int).astype(str)
            + "_"
            + q.tileN.astype(int).astype(str)
        ).isin(held)
    ]
    print(
        f"{len(q):,} stem-years to crop: {q.label.value_counts().to_dict()}; "
        f"{q.dest.nunique():,} RGB tile-years",
        flush=True,
    )
    if args.limit:
        q = q[q.dest.isin(q.dest.drop_duplicates().sample(args.limit, random_state=0))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = df_main.deepforest()
    detector.load_model("weecology/deepforest-tree")
    model = detector.model.to(device).eval()
    for lab in ("Alive", "Dead"):
        (args.root / "crops" / "neon_vst" / lab).mkdir(parents=True, exist_ok=True)

    records = []
    for n, ((rgb_path, chm_path), rows) in enumerate(
        q.groupby(["dest", "chm_path"], dropna=False)
    ):
        try:
            with torch.no_grad():
                recs = tile_year_crops(rgb_path, chm_path, rows, model, args, device)
        except rasterio.errors.RasterioIOError as e:
            print("skip", rgb_path, e, flush=True)
            continue
        for r in recs:
            rel = (
                Path("crops")
                / "neon_vst"
                / r["label"]
                / f"{r['individualID']}_{r['year']}.png"
            )
            Image.fromarray(r.pop("crop")).save(args.root / rel)
            records.append({"crop_path": str(rel), **r})
        if n % 50 == 0:
            print(f"{n} tile-years, {len(records):,} crops", flush=True)

    df = pd.DataFrame(records).assign(source="neon_vst", label_type="weak")
    df["width"] = df.xmax - df.xmin
    df["height"] = df.ymax - df.ymin
    df = flag(df, args)
    if args.score_model:
        df["p_dead"] = score(df, args.root, args.score_model)
    suffix = "_test" if args.limit else ""
    out = args.root / "manifests"
    df.to_csv(out / f"neon_vst_unfiltered{suffix}.csv", index=False)
    kept = df[~df.filter(like="flag_").any(axis=1)]
    kept.to_csv(out / f"neon_vst{suffix}.csv", index=False)
    print("flags:", df.filter(like="flag_").groupby(df.label).sum().to_dict())
    print(
        "kept:",
        kept.label.value_counts().to_dict(),
        "individuals",
        kept.individualID.nunique(),
        "groups",
        kept.group.nunique(),
    )
    if "p_dead" in kept:
        print(
            "p_dead by label:\n",
            kept.groupby("label").p_dead.describe().round(3).to_string(),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--release", default="RELEASE-2026")
    p.add_argument("--holdout-split", default="neon_hand_v1")
    p.add_argument(
        "--max-years-after",
        type=int,
        default=5,
        help="skip flights this many years past the last field visit",
    )
    p.add_argument(
        "--window-m", type=float, default=40.0, help="detection window around stems"
    )
    p.add_argument("--min-score", type=float, default=0.3)
    p.add_argument(
        "--max-dist-m", type=float, default=4.0, help="max stem to crown-centre distance"
    )
    p.add_argument("--height-tol-m", type=float, default=7.0)
    p.add_argument("--height-tol-frac", type=float, default=0.75)
    p.add_argument(
        "--pad", type=int, default=10, help="px added around each crown (10 px = 1 m)"
    )
    p.add_argument("--min-side-m", type=float, default=1.5)
    p.add_argument("--max-aspect", type=float, default=2.5)
    p.add_argument(
        "--score-model", default="", help="CropModel checkpoint or HF id for p_dead"
    )
    p.add_argument(
        "--limit", type=int, default=0, help="only N random RGB tile-years (testing)"
    )
    main(p.parse_args())
