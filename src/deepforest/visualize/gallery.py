"""Utilities to export detection crops for visualization.

This module provides helpers to export thumbnails from detection results
and create HTML galleries for local viewing.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image


def _resolve_image_path(row: Any, root_dir: str | None = None) -> str | None:
    """Resolve an image path from a result row.

    The row may be a mapping (dict-like) or a pandas Series with an
    ``image_path`` entry. If a ``root_dir`` is provided and the path is
    relative, the function returns an absolute path joined with the
    root.
    """

    image_path: str | None = None
    if isinstance(row, dict):
        image_path = row.get("image_path")
    else:
        image_path = (
            row.get("image_path") if "image_path" in getattr(row, "index", []) else None
        )

    if image_path and root_dir:
        p = Path(image_path)
        if not p.is_absolute():
            image_path = str(Path(root_dir) / image_path)

    return image_path


def _sanitize_label(label: str | None) -> str:
    """Create a filesystem-safe, short label from a value."""
    if label is None:
        return "unknown"
    s = str(label).strip().lower()
    s = "".join(c if c.isalnum() or c in "-_" else "_" for c in s)
    return s or "unknown"


def export_to_gallery(
    results_df: Any,
    savedir: str,
    root_dir: str | None = None,
    padding: int = 8,
    thumb_size: tuple[int, int] = (256, 256),
    max_crops: int | None = None,
    sample_seed: int | None = None,
    point_size: int = 64,
    logger: logging.Logger | None = None,
    sample_by_image: bool = False,
    per_image_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Export crops from a results DataFrame into a static gallery.

    Args:
        results_df: pandas DataFrame with prediction rows. Rows must include either
            xmin,ymin,xmax,ymax columns or a `geometry` with `.bounds`. Each row
            should include `image_path` (absolute or relative) or `results_df.root_dir`
            can be set and will be used to resolve relative paths.
        savedir: directory to write thumbnails and metadata into.
        root_dir: optional root to resolve relative image paths.
        padding: pixels to pad around bounding boxes.
        thumb_size: (width, height) of output thumbnails.
        max_crops: maximum number of crops to export (sampling applied if fewer rows exist).
        sample_seed: deterministic seed for sampling when max_crops < available crops.
        point_size: box size (pixels) for point annotations.

    Writes:
        savedir/thumbnails/*.jpg
        savedir/metadata.json
        savedir/metadata.csv

    Returns:
        metadata (list of dict)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    savedir_p = Path(savedir)
    savedir_p.mkdir(parents=True, exist_ok=True)
    thumbs_dir = savedir_p / "thumbnails"
    thumbs_dir.mkdir(exist_ok=True)

    # Build list of candidate indices that have image_path and bbox info
    # Optionally sample by image to ensure broad coverage across images.
    img_to_indices: dict[str, list[int]] = {}
    for i, row in results_df.iterrows():
        img_path = _resolve_image_path(
            row, root_dir or getattr(results_df, "root_dir", None)
        )
        if not img_path:
            continue
        img_to_indices.setdefault(img_path, []).append(i)

    # Prepare sampling
    image_paths = list(img_to_indices.keys())
    if sample_seed is not None:
        rng = np.random.RandomState(sample_seed)
        rng.shuffle(image_paths)

    indices: list[int] = []
    if sample_by_image:
        # If per_image_limit is provided, take up to that many rows per image (fast path).
        if per_image_limit is not None:
            for img in image_paths:
                rows = img_to_indices[img][:]
                if sample_seed is not None:
                    rng.shuffle(rows)
                selected = rows[:per_image_limit]
                indices.extend(selected)
                if max_crops is not None and len(indices) >= max_crops:
                    indices = indices[:max_crops]
                    break
        else:
            # Round-robin across images to distribute crops evenly.
            # Prepare per-image shuffled queues
            queues = []
            for img in image_paths:
                rows = img_to_indices[img][:]
                if sample_seed is not None:
                    rng.shuffle(rows)
                queues.append(list(rows))

            # Round-robin selection
            more = True
            while more and (max_crops is None or len(indices) < max_crops):
                more = False
                for q in queues:
                    if not q:
                        continue
                    more = True
                    indices.append(q.pop(0))
                    if max_crops is not None and len(indices) >= max_crops:
                        break
    else:
        # flatten per-row sampling
        for img in image_paths:
            rows = img_to_indices[img]
            indices.extend(rows)
        if sample_seed is not None:
            rng.shuffle(indices)
        if max_crops is not None:
            indices = indices[:max_crops]

    metadata: list[dict[str, Any]] = []
    written = 0
    for idx in indices:
        row = results_df.loc[idx]
        img_path = _resolve_image_path(
            row, root_dir or getattr(results_df, "root_dir", None)
        )
        if not img_path or not Path(img_path).exists():
            logger.warning("Image not found, skipping: %s", img_path)
            continue

        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                # extract bbox
                xmin = ymin = xmax = ymax = None
                for names in [
                    ("xmin", "ymin", "xmax", "ymax"),
                    ("x_min", "y_min", "x_max", "y_max"),
                ]:
                    if all(n in row.index for n in names):
                        xmin, ymin, xmax, ymax = (float(row[names[i]]) for i in range(4))
                        break

                # geometry bounds fallback
                if xmin is None and "geometry" in row.index:
                    geom = row["geometry"]
                    if hasattr(geom, "bounds"):
                        b = geom.bounds
                        xmin, ymin, xmax, ymax = (
                            float(b[0]),
                            float(b[1]),
                            float(b[2]),
                            float(b[3]),
                        )
                    elif hasattr(geom, "x") and hasattr(geom, "y"):
                        # point geometry
                        cx, cy = float(geom.x), float(geom.y)
                        half = point_size / 2.0
                        xmin, ymin, xmax, ymax = (
                            cx - half,
                            cy - half,
                            cx + half,
                            cy + half,
                        )

                # fallback for point columns
                if xmin is None and all(k in row.index for k in ("x", "y")):
                    cx, cy = float(row["x"]), float(row["y"])
                    half = point_size / 2.0
                    xmin, ymin, xmax, ymax = cx - half, cy - half, cx + half, cy + half

                if xmin is None:
                    logger.warning("No bounding box info for row %s, skipping", idx)
                    continue

                # apply padding and clip
                xmin = max(int(math.floor(xmin - padding)), 0)
                ymin = max(int(math.floor(ymin - padding)), 0)
                xmax = min(int(math.ceil(xmax + padding)), im.width)
                ymax = min(int(math.ceil(ymax + padding)), im.height)

                if xmax <= xmin or ymax <= ymin:
                    logger.warning("Empty crop for row %s, skipping", idx)
                    continue

                crop = im.crop((xmin, ymin, xmax, ymax))
                # resize to fixed thumb size
                crop = crop.resize((thumb_size[0], thumb_size[1]), resample=Image.LANCZOS)

                label = None
                if "label" in getattr(row, "index", []):
                    label = row.get("label")
                label_safe = _sanitize_label(label)

                unique = uuid4().hex[:8]
                fname = f"{written:06d}_{label_safe}_{unique}.jpg"
                out_path = thumbs_dir / fname
                crop.save(str(out_path), quality=90)

                meta = {
                    "crop_id": written,
                    "filename": f"thumbnails/{fname}",  # Use forward slashes for web compatibility
                    "source_image": img_path,
                    "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)],
                    "label": label if label is not None else "Unknown",
                    "score": float(row.get("score"))
                    if (
                        "score" in getattr(row, "index", [])
                        and row.get("score") is not None
                    )
                    else None,
                    "width": im.width,
                    "height": im.height,
                }
                metadata.append(meta)
                written += 1
        except Exception as e:
            logger.warning("Error processing %s: %s", img_path, e)
            continue

    # write metadata files
    with open(savedir_p / "metadata.json", "w", encoding="utf8") as fh:
        json.dump(metadata, fh, indent=2)

    # write CSV for convenience
    if metadata:
        csv_path = savedir_p / "metadata.csv"
        with open(csv_path, "w", newline="", encoding="utf8") as csvfile:
            writer = csv.writer(csvfile)
            header = [
                "crop_id",
                "filename",
                "source_image",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "label",
                "score",
                "width",
                "height",
            ]
            writer.writerow(header)
            for m in metadata:
                xmin, ymin, xmax, ymax = m["bbox"]
                writer.writerow(
                    [
                        m["crop_id"],
                        m["filename"],
                        m["source_image"],
                        xmin,
                        ymin,
                        xmax,
                        ymax,
                        m.get("label"),
                        m.get("score"),
                        m.get("width"),
                        m.get("height"),
                    ]
                )

    return metadata


def write_gallery_html(
    savedir: str,
    title: str = "DeepForest Gallery",
    page_size: int = 200,
    embed_images: bool = True,
) -> None:
    """Write a self-contained index.html that displays the thumbnails and
    metadata.

    Args:
        savedir: Directory containing metadata.json and thumbnails/
        title: HTML page title
        page_size: Number of thumbnails to render per "Load more" click
        embed_images: If True, embed images as base64 to avoid CORS issues
    """
    import base64

    savedir_path = Path(savedir)
    metadata_file = savedir_path / "metadata.json"

    # Read metadata
    if not metadata_file.exists():
        raise FileNotFoundError(f"metadata.json not found in {savedir}")

    with open(metadata_file) as f:
        metadata = json.load(f)

    # Optionally embed images as base64 to avoid CORS issues
    if embed_images:
        for item in metadata:
            filename = item.get("filename", "")
            if filename:
                img_path = savedir_path / filename
                if img_path.exists():
                    with open(img_path, "rb") as f:
                        img_data = f.read()
                    img_base64 = base64.b64encode(img_data).decode("utf-8")
                    item["data_url"] = f"data:image/jpeg;base64,{img_base64}"
                else:
                    item["data_url"] = None

    # Create self-contained HTML
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{
        font-family: Arial, sans-serif;
        padding: 1rem;
        background-color: #f5f5f5;
    }}
    .header {{
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .filters {{
        margin-bottom: 12px;
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        align-items: center;
    }}
    .filters input, .filters select {{
        padding: 8px;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 14px;
        min-width: 120px;
    }}
    .filter-group {{
        display: flex;
        flex-direction: column;
        gap: 4px;
    }}
    .filter-group label {{
        font-size: 12px;
        font-weight: bold;
        color: #555;
    }}
    .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill,minmax(200px,1fr));
        grid-gap: 16px;
    }}
    .card {{
        border: 1px solid #ddd;
        padding: 8px;
        border-radius: 8px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }}
    .card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }}
    .card img {{
        width: 100%;
        height: auto;
        display: block;
        border-radius: 4px;
    }}
    .meta {{
        font-size: 12px;
        color: #666;
        margin-top: 8px;
        padding: 4px;
        background: #f8f9fa;
        border-radius: 4px;
    }}
    .load-more {{
        text-align: center;
        margin-top: 16px;
    }}
    .load-more button {{
        padding: 10px 20px;
        background: #007bff;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }}
    .load-more button:hover {{
        background: #0056b3;
    }}
    .stats {{
        font-size: 14px;
        color: #666;
        margin-top: 8px;
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>{title}</h1>
    <p><strong>Multi-Image Gallery View</strong> - Interactive exploration of detection crops from multiple images</p>
    <div class="filters">
      <div class="filter-group">
        <label>Label Filter</label>
        <input id="labelFilter" placeholder="Enter label to filter (e.g., tree, shrub)">
      </div>
      <div class="filter-group">
        <label>Sort By</label>
        <select id="sortBy">
          <option value="crop_id">Crop ID</option>
          <option value="label">Label</option>
          <option value="score">Confidence Score</option>
          <option value="source_image">Source Image</option>
        </select>
      </div>
      <div class="filter-group">
        <label>Min Score</label>
        <input id="minScore" type="number" min="0" max="1" step="0.1" placeholder="0.0">
      </div>
    </div>
    <div class="stats">
      Showing <span id="showing-count">{len(metadata)}</span> of {len(metadata)} items
      | <span id="unique-images">0</span> unique source images
      | <span id="unique-labels">0</span> unique labels
    </div>
  </div>

  <div id="grid" class="grid"></div>

  <div class="load-more">
    <button id="loadMore">Load more</button>
  </div>

  <script>
    // Embedded metadata - avoids CORS issues
    window._galleryData = {json.dumps(metadata, indent=2)};
    window._pageIndex = 0;
    window._pageSize = {page_size};

    function renderItem(item) {{
        const card = document.createElement('div');
        card.className = 'card';

        const img = document.createElement('img');
        if (item.data_url) {{
            img.src = item.data_url;
        }} else if (item.filename) {{
            img.src = item.filename;
        }} else {{
            img.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPk5vIGltYWdlPC90ZXh0Pjwvc3ZnPg==';
        }}
        img.alt = `${{item.label}} detection`;
        card.appendChild(img);

        const meta = document.createElement('div');
        meta.className = 'meta';
        const score = item.score ? item.score.toFixed(3) : 'N/A';
        const sourceImg = item.source_image ? item.source_image.split('/').pop() : 'Unknown';
        meta.innerHTML = `
            <strong>${{item.label}}</strong><br>
            ID: ${{item.crop_id}} | Score: ${{score}}<br>
            <small>Source: ${{sourceImg}}</small>
        `;
        card.appendChild(meta);

        return card;
    }}

    function applyFiltersAndSort() {{
        const data = window._galleryData || [];
        const labelFilter = document.getElementById('labelFilter').value.toLowerCase();
        const minScore = parseFloat(document.getElementById('minScore').value) || 0;
        const sortBy = document.getElementById('sortBy').value;

        // Apply filters
        let filteredData = data.filter(item => {{
            const labelMatch = !labelFilter || (item.label && item.label.toLowerCase().includes(labelFilter));
            const scoreMatch = !item.score || item.score >= minScore;
            return labelMatch && scoreMatch;
        }});

        // Apply sorting
        filteredData.sort((a, b) => {{
            let aVal = a[sortBy];
            let bVal = b[sortBy];

            if (sortBy === 'score') {{
                return (bVal || 0) - (aVal || 0); // Descending for scores
            }}

            if (typeof aVal === 'string') aVal = aVal.toLowerCase();
            if (typeof bVal === 'string') bVal = bVal.toLowerCase();

            return aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
        }});

        return filteredData;
    }}

    function updateStats(filteredData) {{
        const uniqueImages = new Set(filteredData.map(item => item.source_image)).size;
        const uniqueLabels = new Set(filteredData.map(item => item.label)).size;

        document.getElementById('unique-images').textContent = uniqueImages;
        document.getElementById('unique-labels').textContent = uniqueLabels;
    }}

    function renderPage() {{
        const grid = document.getElementById('grid');
        const filteredData = applyFiltersAndSort();

        const start = window._pageIndex * window._pageSize;
        const end = Math.min(start + window._pageSize, filteredData.length);

        for (let i = start; i < end; i++) {{
            const item = filteredData[i];
            grid.appendChild(renderItem(item));
        }}

        const showingCount = Math.min(end, filteredData.length);
        document.getElementById('showing-count').textContent =
            `${{showingCount}} of ${{filteredData.length}}`;

        updateStats(filteredData);
        window._pageIndex += 1;

        if (end >= filteredData.length) {{
            document.getElementById('loadMore').style.display = 'none';
        }} else {{
            document.getElementById('loadMore').style.display = 'inline-block';
        }}
    }}

    function resetGallery() {{
        const grid = document.getElementById('grid');
        grid.innerHTML = '';
        window._pageIndex = 0;
        document.getElementById('loadMore').style.display = 'inline-block';
        renderPage();
    }}

    document.getElementById('labelFilter').addEventListener('input', resetGallery);
    document.getElementById('loadMore').addEventListener('click', renderPage);

    // Initialize
    renderPage();
  </script>
</body>
</html>"""

    with open(savedir_path / "index.html", "w", encoding="utf8") as fh:
        fh.write(html)
