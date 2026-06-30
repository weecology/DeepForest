"""Map metadata-only class priors from a metadata-enabled CropModel checkpoint.

This script visualizes what the spatial-temporal embedding branch contributes
to each class, independent of image content. It evaluates a coarse lat/lon grid
for one or more dates, then writes CSV score rasters and PNG maps.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.transform import from_origin

from deepforest.model import CropModel

try:
    import contextily as ctx
except ImportError:  # pragma: no cover - contextily is an optional visual enhancement.
    ctx = None


SPECIES_ALIASES = {
    "Northern Gannet": "Morus bassanus",
    "Common Eider": "Somateria mollissima",
}

DEFAULT_SPECIES = ["Morus bassanus", "Somateria mollissima"]
DEFAULT_DATES = ["2024-01-15", "2024-04-15", "2024-07-15", "2024-10-15"]
DEFAULT_BOUNDS = (-98.0, 18.0, -55.0, 48.0)  # Gulf of Mexico + western Atlantic


def day_of_year(date: str) -> float:
    return float(dt.datetime.strptime(date, "%Y-%m-%d").timetuple().tm_yday)


def resolve_species(species: list[str]) -> list[str]:
    return [SPECIES_ALIASES.get(name, name) for name in species]


def make_grid(
    bounds: tuple[float, float, float, float], cell_degrees: float
) -> pd.DataFrame:
    min_lon, min_lat, max_lon, max_lat = bounds
    lons = np.arange(min_lon + cell_degrees / 2, max_lon, cell_degrees)
    lats = np.arange(min_lat + cell_degrees / 2, max_lat, cell_degrees)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return pd.DataFrame(
        {
            "lon": lon_grid.ravel(),
            "lat": lat_grid.ravel(),
        }
    )


def load_metadata_model(checkpoint: str, device: str) -> CropModel:
    model = CropModel.load_from_checkpoint(checkpoint, map_location=device)
    model.eval()
    model.to(device)
    if (
        getattr(model, "metadata_encoder", None) is None
        or getattr(model, "classifier", None) is None
    ):
        raise ValueError(
            "Checkpoint is not metadata-enabled. Expected CropModel.metadata_encoder "
            "and CropModel.classifier."
        )
    return model


def metadata_prior_scores(
    model: CropModel,
    grid: pd.DataFrame,
    date: str,
    device: str,
) -> pd.DataFrame:
    """Compute metadata-only logits and probabilities for every grid cell/class."""
    metadata = torch.tensor(
        np.column_stack(
            [
                grid["lat"].to_numpy(),
                grid["lon"].to_numpy(),
                np.full(len(grid), day_of_year(date)),
            ]
        ),
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        meta_features = model.metadata_encoder(metadata)
        meta_dim = meta_features.shape[1]
        classifier = model.classifier
        meta_weights = classifier.weight[:, -meta_dim:]
        logits = meta_features @ meta_weights.T
        if classifier.bias is not None:
            logits = logits + classifier.bias
        probabilities = torch.softmax(logits, dim=1)

    labels = model.numeric_to_label_dict
    rows = []
    logits_np = logits.cpu().numpy()
    probs_np = probabilities.cpu().numpy()
    for class_idx, label in labels.items():
        class_scores = pd.DataFrame(
            {
                "date": date,
                "class_idx": class_idx,
                "species": label,
                "lat": grid["lat"].to_numpy(),
                "lon": grid["lon"].to_numpy(),
                "metadata_logit": logits_np[:, class_idx],
                "metadata_probability": probs_np[:, class_idx],
            }
        )
        rows.append(class_scores)
    return pd.concat(rows, ignore_index=True)


def select_species_scores(scores: pd.DataFrame, species: list[str]) -> pd.DataFrame:
    available = set(scores["species"].unique())
    missing = [name for name in species if name not in available]
    if missing:
        examples = sorted(available)[:20]
        raise ValueError(
            f"Species not found in checkpoint label_dict: {missing}. "
            f"First available labels: {examples}"
        )

    selected = scores[scores["species"].isin(species)].copy()
    selected["relative_score"] = selected.groupby(["date", "species"])[
        "metadata_logit"
    ].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.0
    )
    return selected


def _safe_name(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_")


def plot_species_map(
    scores: pd.DataFrame,
    species: str,
    date: str,
    bounds: tuple[float, float, float, float],
    output_path: Path,
    plot_column: str,
    cell_degrees: float,
    cmap: str,
    use_basemap: bool,
) -> None:
    subset = scores[(scores["species"] == species) & (scores["date"] == date)]
    pivot = subset.pivot(index="lat", columns="lon", values=plot_column).sort_index()
    min_lon, min_lat, max_lon, max_lat = bounds

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_aspect("equal")

    if use_basemap and ctx is not None:
        try:
            ctx.add_basemap(
                ax,
                crs="EPSG:4326",
                source=ctx.providers.Esri.OceanBasemap,
                attribution_size=5,
                zorder=0,
            )
        except Exception as exc:
            print(f"Could not add basemap tiles: {exc}")

    image = ax.imshow(
        pivot.to_numpy(),
        extent=[
            pivot.columns.min() - cell_degrees / 2,
            pivot.columns.max() + cell_degrees / 2,
            pivot.index.min() - cell_degrees / 2,
            pivot.index.max() + cell_degrees / 2,
        ],
        origin="lower",
        cmap=cmap,
        alpha=0.75,
        zorder=2,
        vmin=0 if plot_column == "relative_score" else None,
        vmax=1 if plot_column == "relative_score" else None,
    )
    fig.colorbar(image, ax=ax, label=plot_column.replace("_", " "))
    ax.set_title(f"{species} metadata prior, {date}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(color="white", linewidth=0.3, alpha=0.4)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def write_species_geotiff(
    scores: pd.DataFrame,
    species: str,
    date: str,
    output_path: Path,
    plot_column: str,
    cell_degrees: float,
) -> None:
    subset = scores[(scores["species"] == species) & (scores["date"] == date)]
    pivot = subset.pivot(index="lat", columns="lon", values=plot_column).sort_index()
    array = np.flipud(pivot.to_numpy()).astype("float32")
    transform = from_origin(
        pivot.columns.min() - cell_degrees / 2,
        pivot.index.max() + cell_degrees / 2,
        cell_degrees,
        cell_degrees,
    )
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(array, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize metadata-only species priors from a CropModel checkpoint."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Metadata-enabled CropModel checkpoint."
    )
    parser.add_argument(
        "--species",
        nargs="+",
        default=DEFAULT_SPECIES,
        help="Scientific names to map. Common aliases supported: Northern Gannet, Common Eider.",
    )
    parser.add_argument(
        "--dates", nargs="+", default=DEFAULT_DATES, help="YYYY-MM-DD dates to map."
    )
    parser.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        default=DEFAULT_BOUNDS,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
    )
    parser.add_argument(
        "--cell-degrees", type=float, default=1.0, help="Grid cell size in degrees."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/metadata_prior_maps")
    )
    parser.add_argument(
        "--plot-column",
        default="relative_score",
        choices=["relative_score", "metadata_probability", "metadata_logit"],
        help="Score column used for PNG coloring. CSV always contains all score columns.",
    )
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-basemap", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    species = resolve_species(args.species)
    grid = make_grid(tuple(args.bounds), args.cell_degrees)
    model = load_metadata_model(args.checkpoint, args.device)

    all_scores = []
    for date in args.dates:
        scores = metadata_prior_scores(
            model=model, grid=grid, date=date, device=args.device
        )
        selected = select_species_scores(scores, species)
        all_scores.append(selected)

        for species_name in species:
            output_stem = args.output_dir / f"{_safe_name(species_name)}_{date}"
            plot_species_map(
                scores=selected,
                species=species_name,
                date=date,
                bounds=tuple(args.bounds),
                output_path=output_stem.with_suffix(".png"),
                plot_column=args.plot_column,
                cell_degrees=args.cell_degrees,
                cmap=args.cmap,
                use_basemap=not args.no_basemap,
            )
            write_species_geotiff(
                scores=selected,
                species=species_name,
                date=date,
                output_path=output_stem.with_suffix(".tif"),
                plot_column=args.plot_column,
                cell_degrees=args.cell_degrees,
            )
            print(f"Wrote {output_stem.with_suffix('.png')}")
            print(f"Wrote {output_stem.with_suffix('.tif')}")

    combined = pd.concat(all_scores, ignore_index=True)
    csv_path = args.output_dir / "metadata_prior_scores.csv"
    combined.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
