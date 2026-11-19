"""Cut SeaBee seabird orthomosaics into bird-containing tiles for DeepForest.

Annotations come from Zenodo 17379310 (lon/lat polygons). Imagery is
fetched from the SeaBee GeoNode WCS one tile-window at a time, only for
windows that contain birds, so full mosaics are never stored. Output is
a DeepForest CSV (Pascal VOC pixel boxes).
"""

import argparse
import io
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image
from pyproj import Transformer
from shapely import wkb

OWS = "https://geonode.seabee.sigma2.no/geoserver/ows"
MERGE = {350: 9050, 360: 9050}  # cormorant/shag grouping used by the SeaBee authors


def describe(layer: str) -> dict:
    """Return CRS, native extent and pixel size of a GeoNode coverage."""
    cid = layer.replace(":", "__")
    xml = requests.get(
        OWS,
        params={
            "service": "WCS",
            "version": "2.0.1",
            "request": "DescribeCoverage",
            "coverageid": cid,
        },
        timeout=60,
    ).text
    epsg = int(re.search(r"EPSG/0/(\d+)", xml).group(1))
    lo = [float(v) for v in re.search(r"lowerCorner>([^<]+)<", xml).group(1).split()]
    hi = [float(v) for v in re.search(r"upperCorner>([^<]+)<", xml).group(1).split()]
    high = [int(v) for v in re.search(r"<gml:high>([^<]+)<", xml).group(1).split()]
    res = (hi[0] - lo[0]) / (high[0] + 1)
    return {"cid": cid, "epsg": epsg, "lo": lo, "hi": hi, "res": res}


def fetch_tile(layer: str, epsg: int, x0: float, y0: float, size: float, px: int):
    """Fetch a square window via WMS at px x px pixels; return RGBA array or
    None."""
    params = {
        "service": "WMS",
        "version": "1.1.1",
        "request": "GetMap",
        "layers": layer,
        "styles": "",
        "format": "image/png",
        "transparent": "true",
        "width": px,
        "height": px,
        "srs": f"EPSG:{epsg}",
        "bbox": f"{x0},{y0},{x0 + size},{y0 + size}",
    }
    for _ in range(3):
        try:
            r = requests.get(OWS, params=params, timeout=300)
            if r.ok and r.headers.get("content-type", "").startswith("image"):
                return np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))
        except requests.RequestException:
            pass
    return None


def plan_mosaic(args, item):
    """Project annotations for one mosaic and group them into tile windows."""
    name, ann, layer = item
    info = describe(layer)
    proj = Transformer.from_crs(4326, info["epsg"], always_xy=True)
    bounds = np.array([wkb.loads(g, hex=True).bounds for g in ann.geom])
    xa, ya = proj.transform(bounds[:, 0], bounds[:, 1])
    xb, yb = proj.transform(bounds[:, 2], bounds[:, 3])
    ann = ann.assign(
        X0=np.minimum(xa, xb),
        X1=np.maximum(xa, xb),
        Y0=np.minimum(ya, yb),
        Y1=np.maximum(ya, yb),
    )
    size = args.patch_size * info["res"]
    ann["ci"] = np.floor(((ann.X0 + ann.X1) / 2 - info["lo"][0]) / size).astype(int)
    ann["cj"] = np.floor(((ann.Y0 + ann.Y1) / 2 - info["lo"][1]) / size).astype(int)
    return [
        (name, layer, info, size, ci, cj, g) for (ci, cj), g in ann.groupby(["ci", "cj"])
    ]


def process_tile(args, job):
    """Fetch one tile window, save the PNG and return its boxes in pixel
    coordinates."""
    name, layer, info, size, ci, cj, grp = job
    png = Path(args.out_dir) / "tiles" / f"seabee_{name}_{ci}_{cj}.png"
    tx0, ty0 = info["lo"][0] + ci * size, info["lo"][1] + cj * size
    tile = fetch_tile(layer, info["epsg"], tx0, ty0, size, args.patch_size)
    if tile is None or (tile[..., 3] > 0).mean() < args.min_valid:
        return None
    n = args.patch_size
    # WMS window is exactly [tx0, tx0+size] x [ty0, ty0+size]
    px0, px1 = (grp.X0 - tx0) / info["res"], (grp.X1 - tx0) / info["res"]
    py0, py1 = (ty0 + size - grp.Y1) / info["res"], (ty0 + size - grp.Y0) / info["res"]
    area = (px1 - px0) * (py1 - py0)
    cx0, cx1, cy0, cy1 = px0.clip(0, n), px1.clip(0, n), py0.clip(0, n), py1.clip(0, n)
    keep = ((cx1 - cx0) * (cy1 - cy0)) >= args.min_overlap * area
    if not keep.any():
        return None
    Image.fromarray(tile[..., :3]).save(png)
    return pd.DataFrame(
        {
            "image_path": f"seabee_tiles/{png.name}",
            "xmin": cx0[keep],
            "ymin": cy0[keep],
            "xmax": cx1[keep],
            "ymax": cy1[keep],
            "label": "Bird",
            "species": grp.species[keep],
            "mosaic": name,
        }
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zenodo-dir", required=True, help="extracted Zenodo folder")
    p.add_argument("--mosaic-map", required=True, help="CSV: filename,alternate")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--patch-size", type=int, default=1000)
    p.add_argument("--min-overlap", type=float, default=0.5)
    p.add_argument(
        "--min-valid", type=float, default=0.1, help="min non-transparent fraction"
    )
    p.add_argument(
        "--test-box-frac", type=float, default=0.12, help="approx share of boxes held out"
    )
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="only first N mosaics")
    args = p.parse_args()

    (Path(args.out_dir) / "tiles").mkdir(parents=True, exist_ok=True)
    ann = pd.read_csv(Path(args.zenodo_dir) / "Data" / "annotations.csv")
    ann["species"] = ann.species.replace(MERGE)
    mm = pd.read_csv(args.mosaic_map).dropna(subset=["alternate"])
    ann = ann.merge(mm[["filename", "alternate"]], on="filename")
    items = [
        (
            re.sub(r"\W+", "_", f.split("/")[1] + "_" + f.split("/")[2]),
            g,
            g.alternate.iloc[0],
        )
        for f, g in ann.groupby("filename")
    ][: args.limit]

    with ThreadPoolExecutor(args.workers) as ex:
        jobs = [j for js in ex.map(lambda i: plan_mosaic(args, i), items) for j in js]
        print(f"{len(jobs)} tile windows", flush=True)
        res = [r for r in ex.map(lambda j: process_tile(args, j), jobs) if r is not None]
    df = pd.concat(res, ignore_index=True)

    # Hold out whole mosaics (random order) until ~test_box_frac of boxes are in test
    per = df.groupby("mosaic").size().sample(frac=1, random_state=args.seed)
    test = set(per.index[per.cumsum() <= args.test_box_frac * len(df)])
    df["split"] = np.where(df.mosaic.isin(test), "test", "train")
    cols = ["image_path", "xmin", "ymin", "xmax", "ymax", "label"]
    for s in ("train", "test"):
        df[df.split == s][cols].to_csv(Path(args.out_dir) / f"{s}.csv", index=False)
    df.to_csv(Path(args.out_dir) / "all_with_metadata.csv", index=False)
    print(
        df.groupby("split").agg(
            boxes=("label", "size"),
            tiles=("image_path", "nunique"),
            mosaics=("mosaic", "nunique"),
        )
    )


if __name__ == "__main__":
    main()
