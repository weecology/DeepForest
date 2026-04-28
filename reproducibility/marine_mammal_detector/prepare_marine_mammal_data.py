"""Filter BOEM/USGS detection CSVs to marine mammal examples only.

This script expects train/test/zero-shot CSVs generated upstream (e.g. from BOEM
HPC data preparation) and writes DeepForest-ready CSVs for a single-class
"Object" detector.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_PATTERNS = [
    "seal",
    "sea lion",
    "sealion",
    "walrus",
    "dolphin",
    "whale",
    "porpoise",
    "otter",
    "manatee",
    "dugong",
]


def _is_empty_row(frame: pd.DataFrame) -> pd.Series:
    return (
        (frame["xmin"] == 0)
        & (frame["xmax"] == 0)
        & (frame["ymin"] == 0)
        & (frame["ymax"] == 0)
    )


def _contains_any_pattern(values: pd.Series, patterns: list[str]) -> pd.Series:
    normalized = values.fillna("").astype(str).str.lower()
    matches = pd.Series(False, index=values.index)
    for pattern in patterns:
        matches = matches | normalized.str.contains(pattern.lower(), regex=False)
    return matches


def filter_marine_rows(frame: pd.DataFrame, patterns: list[str]) -> pd.DataFrame:
    """Keep rows that are marine mammal labels or empty-image rows."""
    required = {"image_path", "xmin", "ymin", "xmax", "ymax", "label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    empty_mask = _is_empty_row(frame)
    marine_mask = _contains_any_pattern(frame["label"], patterns)
    keep = frame[empty_mask | marine_mask].copy()
    keep["label"] = "Object"
    return keep


def process_file(input_csv: Path, output_csv: Path, patterns: list[str]) -> None:
    data = pd.read_csv(input_csv, low_memory=False)
    filtered = filter_marine_rows(data, patterns)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False)

    empty_rows = int(_is_empty_row(filtered).sum()) if not filtered.empty else 0
    print(
        f"{input_csv.name}: {len(data)} -> {len(filtered)} rows "
        f"(empty rows kept: {empty_rows})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter BOEM/USGS CSVs to marine mammal rows."
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--zero-shot-csv", type=Path, required=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--class-pattern",
        action="append",
        dest="class_patterns",
        default=None,
        help=(
            "Substring pattern for marine mammal labels. Repeat as needed. "
            "Defaults to a broad marine mammal list."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patterns = args.class_patterns or DEFAULT_PATTERNS
    print(f"Using class patterns: {patterns}")

    process_file(args.train_csv, args.output_dir / "train.csv", patterns)
    process_file(args.test_csv, args.output_dir / "test.csv", patterns)

    if args.zero_shot_csv is not None:
        process_file(args.zero_shot_csv, args.output_dir / "zero_shot.csv", patterns)


if __name__ == "__main__":
    main()
