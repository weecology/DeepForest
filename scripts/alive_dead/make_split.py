"""Make a grouped train/val/test split of a crop manifest.

Groups (1 km tiles, orthophotos, ...) are never shared between splits.
StratifiedGroupKFold keeps the Dead fraction similar in each split. One fold is
test, one is val, the rest is train.

The split is written as a CSV and as ImageFolder trees of symlinks
(``splits/<name>/{train,val,test}/{Alive,Dead}``) that ``CropModel.load_from_disk``
reads directly.

Usage:
    python make_split.py --root /blue/ewhite/b.weinstein/DeepForest/alive_dead \
        --manifest neon_hand.csv --name neon_hand_v1 --n-folds 7 --seed 0
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def assign_splits(df: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    """Return 'train', 'val' or 'test' for each row, grouped by ``df.group``."""
    folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    split = pd.Series("train", index=df.index)
    fold_of = {}
    for k, (_, idx) in enumerate(folds.split(df, df.label, df.group)):
        fold_of.update(dict.fromkeys(df.index[idx], k))
    fold = pd.Series(fold_of)
    split[fold == 0] = "test"
    split[fold == 1] = "val"
    return split


def write_imagefolder(df: pd.DataFrame, root: Path, out: Path) -> None:
    """Symlink each crop into out/<split>/<label>/ with a unique file name."""
    for row in df.itertuples():
        dest = out / row.split / row.label / f"{row.source}__{Path(row.crop_path).name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            os.symlink(root / row.crop_path, dest)


def main(
    root: Path, manifests: list[str], name: str, n_folds: int, seed: int, all_test: bool
) -> None:
    df = pd.concat(
        [pd.read_csv(root / "manifests" / m, low_memory=False) for m in manifests],
        ignore_index=True,
    )
    df["group"] = df.source + ":" + df.group.astype(str)
    df["split"] = "test" if all_test else assign_splits(df, n_folds, seed)

    leaks = df.groupby("group").split.nunique().gt(1).sum()
    assert leaks == 0, f"{leaks} groups appear in more than one split"
    for key in ("md5", "dhash"):
        if key in df:
            shared = df.groupby(key).split.nunique().gt(1).sum()
            assert shared == 0, (
                f"{shared} identical images ({key}) appear in more than one split"
            )

    out = root / "splits" / name
    if out.exists():
        raise FileExistsError(f"{out} exists; choose a new --name")
    out.mkdir(parents=True)
    df.to_csv(out / "split.csv", index=False)
    write_imagefolder(df, root, out)

    summary = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    for lab in ("Alive", "Dead"):
        if lab not in summary:
            summary[lab] = 0
    summary["groups"] = df.groupby("split").group.nunique()
    summary["sites"] = df.groupby("split").site.nunique()
    summary["dead_frac"] = (summary.Dead / (summary.Alive + summary.Dead)).round(3)
    summary.to_csv(out / "summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--manifest", required=True, nargs="+", help="CSV file name(s) under manifests/"
    )
    parser.add_argument(
        "--name", required=True, help="split name, written to splits/<name>"
    )
    parser.add_argument("--n-folds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--all-test", action="store_true", help="put every crop in test (benchmarks)"
    )
    a = parser.parse_args()
    main(a.root, a.manifest, a.name, a.n_folds, a.seed, a.all_test)
