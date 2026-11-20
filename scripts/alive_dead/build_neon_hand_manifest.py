"""Build the manifest for the hand-labelled NEON alive/dead crops.

Reads every PNG under ``<root>/sources/neon_hand``. Byte-identical copies are
merged, and images labelled both Alive and Dead are dropped and listed for
review. Each crop gets a spatial group id: the 1 km NEON tile it came from.
Plot crops (``SITE_plot_year_k.png``) are mapped to a tile with the plot's stem
coordinates from NEON Woody Vegetation Structure.

Writes ``manifests/neon_hand.csv`` and ``manifests/neon_hand_conflicts.csv``.

Usage:
    python build_neon_hand_manifest.py --root /blue/ewhite/b.weinstein/DeepForest/alive_dead
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

TILE_RE = r"^(?P<year>\d{4})_(?P<site>[A-Z]{4})_\d+_(?P<easting>\d{6})_(?P<northing>\d{7})_image"
PLOT_RE = r"^(?P<site>[A-Z]{4})_(?P<plotnum>\d{3})_(?P<year>\d{4})_\d+\.png$"


def dhash(path: Path, size: int = 8) -> str:
    """Difference hash, used to find near-duplicate crops."""
    img = Image.open(path).convert("L").resize((size + 1, size))
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.packbits(bits).tobytes().hex()


def scan(root: Path) -> pd.DataFrame:
    """List every crop with its label, sub-source, md5 and dhash."""
    files = sorted((root / "sources" / "neon_hand").rglob("*.png"))
    df = pd.DataFrame({"src_path": [str(f.relative_to(root)) for f in files]})
    df["label"] = [f.parent.name for f in files]
    df["subsource"] = [
        f.relative_to(root / "sources" / "neon_hand").parts[0] for f in files
    ]
    df["orig_split"] = [
        f.parent.parent.name if f.parent.parent.name.startswith("dead_") else ""
        for f in files
    ]
    df["file"] = [f.name for f in files]
    df["md5"] = [hashlib.md5(f.read_bytes()).hexdigest() for f in files]
    df["dhash"] = [dhash(f) for f in files]
    sizes = [Image.open(f).size for f in files]
    df["width"], df["height"] = zip(*sizes, strict=True)
    return df


def plot_tiles(root: Path) -> pd.DataFrame:
    """Map NEON plotID to its 1 km tile from VST stem coordinates."""
    vst = pd.concat(
        pd.read_csv(f, usecols=["plotID", "itcEasting", "itcNorthing"], low_memory=False)
        for f in sorted((root / "sources" / "neon_vst").glob("neon_vst_data_*.csv"))
    )
    c = vst.groupby("plotID")[["itcEasting", "itcNorthing"]].median()
    return pd.DataFrame(
        {
            "easting": (c.itcEasting // 1000 * 1000).astype(int),
            "northing": (c.itcNorthing // 1000 * 1000).astype(int),
        }
    )


def add_location(df: pd.DataFrame, plots: pd.DataFrame) -> pd.DataFrame:
    """Parse site, year and 1 km tile from the file name."""
    tile = df.file.str.extract(TILE_RE)
    plot = df.file.str.extract(PLOT_RE)
    df["site"] = tile.site.fillna(plot.site)
    df["year"] = pd.to_numeric(tile.year.fillna(plot.year))
    df["plot_id"] = (plot.site + "_" + plot.plotnum).where(plot.site.notna())
    plot_xy = plots.reindex(df.plot_id)
    df["easting"] = pd.to_numeric(tile.easting).fillna(
        pd.Series(plot_xy.easting.values, index=df.index)
    )
    df["northing"] = pd.to_numeric(tile.northing).fillna(
        pd.Series(plot_xy.northing.values, index=df.index)
    )
    df["name_format"] = np.where(
        tile.site.notna(), "tile", np.where(plot.site.notna(), "plot", "unknown")
    )
    loc = (
        df.site
        + "_"
        + df.easting.astype("Int64").astype(str)
        + "_"
        + df.northing.astype("Int64").astype(str)
    )
    # Plots with no VST coordinates fall back to their own plot id
    df["group"] = loc.where(df.easting.notna(), df.plot_id)
    return df


def main(root: Path) -> None:
    raw = scan(root)
    raw = add_location(raw, plot_tiles(root))

    # Byte-identical copies with conflicting labels are dropped
    n_labels = raw.groupby("md5").label.nunique()
    conflict = raw.md5.isin(n_labels[n_labels > 1].index)
    conflicts = raw[conflict].sort_values(["md5", "label"])

    # Keep one copy per md5, preferring the git-tracked crop
    kept = raw[~conflict].sort_values(["md5", "subsource"]).drop_duplicates("md5")
    kept = kept.assign(
        source="neon_hand",
        label_type="hand",
        gsd_m=0.1,
        crop_path=kept.src_path,
        n_copies=kept.md5.map(raw.md5.value_counts()),
    )

    # Near-duplicates (same dhash) must not straddle splits, so they share a group
    first_group = kept.groupby("dhash").group.transform("first")
    kept["group"] = first_group

    out = root / "manifests"
    out.mkdir(exist_ok=True)
    cols = [
        "crop_path",
        "label",
        "source",
        "label_type",
        "site",
        "year",
        "group",
        "plot_id",
        "easting",
        "northing",
        "gsd_m",
        "width",
        "height",
        "subsource",
        "orig_split",
        "name_format",
        "md5",
        "dhash",
        "n_copies",
    ]
    kept[cols].to_csv(out / "neon_hand.csv", index=False)
    conflicts[["src_path", "label", "subsource", "orig_split", "md5", "group"]].to_csv(
        out / "neon_hand_conflicts.csv", index=False
    )

    print(f"scanned {len(raw)} files, {raw.md5.nunique()} unique images")
    print(
        f"dropped {conflicts.md5.nunique()} images with conflicting labels ({len(conflicts)} files)"
    )
    print(f"kept {len(kept)}: {kept.label.value_counts().to_dict()}")
    print(
        f"sites {kept.site.nunique()}, groups {kept.group.nunique()}, no group {kept.group.isna().sum()}"
    )
    print(kept.groupby("subsource").label.value_counts().unstack().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="alive_dead data root")
    main(parser.parse_args().root)
