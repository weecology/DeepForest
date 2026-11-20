"""Write review images for the NEON VST crops and zip them for a human pass.

Each image shows a context window around the target crown (box outlined, stem
marked) next to the enlarged crop, with label, year and field data in a header.
Images are named ``<Label>/<crop id>.jpg``, where the crop id is the crop PNG's
stem. The annotator deletes images that are obviously wrong and returns the
folder; ``apply_vst_review.py`` drops the matching manifest rows.

Usage:
    python make_vst_review.py --root /blue/ewhite/b.weinstein/DeepForest/alive_dead
"""

import argparse
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import rasterio
from build_vst_crops import read_rgb
from PIL import Image, ImageDraw, ImageFont
from rasterio.windows import Window

COLORS = {"Dead": (230, 40, 40), "Alive": (40, 220, 80)}
README = """NEON VST weak Alive/Dead crops: quick review
==============================================

Each image shows one target tree crown.
  Left:  ~{ctx} m of context. The target crown is the coloured box (red = labelled Dead,
         green = labelled Alive). The cyan cross is where the field crew mapped the stem.
  Right: the crop that will be used for training, enlarged.
  Header: label, flight year, site, tree ID, field status and height, and the canopy
          height (CHM) inside the box.

Task: DELETE images that are obviously wrong. Keep everything else; don't relabel.
Delete when:
  - Dead folder: the box is a green, live crown (common: the dead tree is under the canopy).
  - Alive folder: the box is clearly a dead/grey crown.
  - The box isn't a tree crown (bare ground, road, shadow, image artefact or black no-data).
When unsure, keep it.

Folders: Dead/ and Alive/. Don't rename or move files.
When done, zip the folders back up and return them.
"""


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


def review_tile(args_tuple) -> int:
    """Write review images for every crop from one RGB tile-year."""
    rgb_path, rows, out_dir, ctx_m, size = args_tuple
    f_big, f_small = font(16), font(13)
    n = 0
    with rasterio.open(rgb_path) as src:
        gsd = src.res[0]
        for r in rows.itertuples():
            cx = (r.crown_xmin + r.crown_xmax) / 2
            cy = (r.crown_ymin + r.crown_ymax) / 2
            half = int(
                max(
                    ctx_m / gsd,
                    1.5 * max(r.crown_xmax - r.crown_xmin, r.crown_ymax - r.crown_ymin),
                )
                / 2
            )
            c0, r0 = int(cx) - half, int(cy) - half
            win = Window(c0, r0, 2 * half, 2 * half)
            ctx = read_rgb(src, win, boundless=True, fill_value=0)
            left = Image.fromarray(ctx).resize((size, size), Image.BILINEAR)
            s = size / (2 * half)
            d = ImageDraw.Draw(left)
            box = [
                (r.crown_xmin - c0) * s,
                (r.crown_ymin - r0) * s,
                (r.crown_xmax - c0) * s,
                (r.crown_ymax - r0) * s,
            ]
            d.rectangle(box, outline=COLORS[r.label], width=3)
            sc, sr = ~src.transform * (r.easting, r.northing)
            px, py = (sc - c0) * s, (sr - r0) * s
            d.line([px - 8, py, px + 8, py], fill=(0, 230, 230), width=2)
            d.line([px, py - 8, px, py + 8], fill=(0, 230, 230), width=2)
            # 5 m scale bar
            bar = 5 / gsd * s
            d.rectangle([10, size - 18, 10 + bar, size - 12], fill=(255, 255, 255))
            d.text((14 + bar, size - 24), "5 m", fill=(255, 255, 255), font=f_small)

            crop = Image.open(r.crop_full).convert("RGB")
            k = size / max(crop.size)
            crop = crop.resize(
                (max(1, int(crop.width * k)), max(1, int(crop.height * k))), Image.NEAREST
            )

            head = 52
            canvas = Image.new("RGB", (2 * size + 10, size + head), (24, 24, 24))
            canvas.paste(left, (0, head))
            canvas.paste(
                crop,
                (size + 10 + (size - crop.width) // 2, head + (size - crop.height) // 2),
            )
            d = ImageDraw.Draw(canvas)
            d.text(
                (8, 6),
                f"{r.label.upper()}  {r.year}  {r.site}  {r.individualID}",
                fill=COLORS[r.label],
                font=f_big,
            )
            chm = "n/a" if pd.isna(r.chm_max) else f"{r.chm_max:.0f} m"
            ht = "n/a" if pd.isna(r.vst_height) else f"{r.vst_height:.0f} m"
            d.text(
                (8, 28),
                f"field: {r.vst_status} ({r.prev_year}), height {ht} | CHM in box {chm} | "
                f"stem-crown {r.match_dist_m:.1f} m",
                fill=(220, 220, 220),
                font=f_small,
            )
            out = out_dir / r.label / f"{Path(r.crop_path).stem}.jpg"
            canvas.save(out, quality=85)
            n += 1
    return n


def main(args: argparse.Namespace) -> None:
    man = pd.read_csv(args.root / "manifests" / f"{args.manifest}.csv", low_memory=False)
    man["crop_full"] = [str(args.root / p) for p in man.crop_path]
    out = args.root / "review" / args.name
    for lab in ("Alive", "Dead"):
        (out / lab).mkdir(parents=True, exist_ok=True)
    (out / "README.txt").write_text(README.format(ctx=args.context_m))
    jobs = [(p, g, out, args.context_m, args.size) for p, g in man.groupby("rgb_path")]
    with ProcessPoolExecutor(args.workers) as pool:
        done = sum(pool.map(review_tile, jobs))
    print(f"{done:,} review images in {out}", flush=True)

    zpath = out.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(out.parent))
    print(f"zip: {zpath} ({zpath.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--manifest", default="neon_vst", help="manifest name under manifests/"
    )
    p.add_argument(
        "--name", default="neon_vst_review_v1", help="review folder and zip name"
    )
    p.add_argument("--context-m", type=float, default=20.0, help="context window width")
    p.add_argument("--size", type=int, default=400, help="px per panel")
    p.add_argument("--workers", type=int, default=8)
    main(p.parse_args())
