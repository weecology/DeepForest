import os
import json
import csv
import pathlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Iterable

import requests
from PIL import Image
from urllib.parse import urlparse, parse_qs

BASE_URL = "http://localhost:8080"
REFRESH_TOKEN = ("your_refresh_token_here")  # Replace with your actual refresh token

TRAIN_DIR = "train_set"
os.makedirs(TRAIN_DIR, exist_ok=True)


def get_access_token() -> str:
    """Obtain a new access token using the refresh token.

    Sends a POST request to the API's token refresh endpoint with the provided refresh token.
    Raises an HTTPError if the request fails.

    Returns:
        str: The newly obtained access token.

    Raises:
        requests.HTTPError: If the HTTP request to refresh the token fails.
    """
    r = requests.post(
        f"{BASE_URL}/api/token/refresh",
        json={"refresh": REFRESH_TOKEN},
        timeout=10,
        headers={"Content-Type": "application/json"},
    )
    r.raise_for_status()
    return r.json()["access"]


def Health_check(access: str) -> Dict[str, str]:
    """Generate an authorization header for Label Studio API requests.

    Args:
        access (str): The access token to be used for authentication.

    Returns:
        Dict[str, str]: A dictionary containing the 'Authorization' header with the provided access token.
    """
    return {"Authorization": f"Bearer {access}"}


def list_projects(access: str) -> List[Dict[str, Any]]:
    """Retrieve a list of projects from the Label Studio API.

    Args:
        access (str): Access token or authentication string for API requests.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a project.

    Raises:
        requests.HTTPError: If the HTTP request to the API fails.
    """
    r = requests.get(f"{BASE_URL}/api/projects?page_size=1000000",
                     headers=Health_check(access),
                     timeout=15)
    r.raise_for_status()
    data = r.json()
    return data["results"] if isinstance(data, dict) and "results" in data else data


def paginate(url: str, headers: Dict[str, str]) -> List[Dict[str, Any]]:
    """Fetches and paginates results from a given API endpoint.

    Args:
        url (str): The initial URL to fetch data from.
        headers (Dict[str, str]): HTTP headers to include in the request.

    Returns:
        List[Dict[str, Any]]: A list of items retrieved from all paginated API responses.

    Raises:
        requests.HTTPError: If an HTTP error occurs during the request.

    Notes:
        - Assumes the API response contains a "results" key for paginated data and a "next" key for the next page URL.
        - If the response is a list or a single dictionary, it is appended directly to the results.
    """
    items: List[Dict[str, Any]] = []
    next_url = url
    while next_url:
        r = requests.get(next_url, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "results" in data:
            items.extend(data["results"])
            next_url = data.get("next")
        else:
            items.extend(data if isinstance(data, list) else [data])
            next_url = None
    return items


def list_tasks(access: str, project_id: int) -> List[Dict[str, Any]]:
    """Retrieve a list of tasks from a Label Studio project, including
    annotations and predictions.

    Args:
        access (str): Access token or credentials for authentication.
        project_id (int): The ID of the Label Studio project to fetch tasks from.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a task with its associated data.
    """
    # fields=all should include annotations & predictions
    url = f"{BASE_URL}/api/tasks?project={project_id}&page_size=200&fields=all"
    return paginate(url, Health_check(access))


def _filter_finished_only(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep tasks that have at least one annotation with a non-empty result."""
    finished = []
    for t in tasks:
        anns = t.get("annotations") or []
        if any((a.get("result") for a in anns)):
            finished.append(t)
    return finished


def ls_json_to_deepforest_csv(
    tasks: Iterable[Dict[str, Any]],
    images_dir: str,
    classes: List[str],
    out_csv: str,
    round_id: Optional[int] = None,
    strict_labels: bool = True,
) -> Tuple[int, int]:
    """Convert Label Studio rectanglelabels annotations to a DeepForest CSV.

    Returns (n_boxes_written, n_unique_images_written).
    """
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    size_cache: Dict[str, Tuple[int, int]] = {}

    def _img_size(p: str) -> Optional[Tuple[int, int]]:
        if p in size_cache:
            return size_cache[p]
        try:
            with Image.open(p) as im:
                size_cache[p] = im.size  # (W, H)
            return size_cache[p]
        except Exception:
            return None

    rows: List[List[Any]] = []
    images_used: set = set()

    for task in tasks:
        data = task.get("data", {}) or {}
        img_field = find_image_field(data)
        if not img_field:
            continue

        # Resolve to a filename under images_dir
        fname = parse_image_url(absolute_image_url(img_field))
        img_path = str(Path(images_dir) / fname)

        size = _img_size(img_path)
        if size is None:
            # Image is not available locally; skip to keep dataset consistent
            continue
        W, H = size

        for ann in task.get("annotations") or []:
            for res in ann.get("result") or []:
                if res.get("type") != "rectanglelabels":
                    continue
                val = res.get("value") or {}
                labs = val.get("rectanglelabels") or []
                if not labs:
                    continue
                label = str(labs[0])
                if strict_labels and label not in classes:
                    continue

                # LS percentages → pixels
                try:
                    x = float(val.get("x", 0.0))
                    y = float(val.get("y", 0.0))
                    w = float(val.get("width", 0.0))
                    h = float(val.get("height", 0.0))
                except Exception:
                    continue

                xmin = int(round((x / 100.0) * W))
                ymin = int(round((y / 100.0) * H))
                xmax = int(round(((x + w) / 100.0) * W))
                ymax = int(round(((y + h) / 100.0) * H))

                # Clamp to image bounds
                xmin = max(0, min(W, xmin))
                ymin = max(0, min(H, ymin))
                xmax = max(0, min(W, xmax))
                ymax = max(0, min(H, ymax))

                # Require positive area
                if xmax <= xmin or ymax <= ymin:
                    continue

                if round_id is None:
                    rows.append([img_path, xmin, ymin, xmax, ymax, label])
                else:
                    rows.append([img_path, xmin, ymin, xmax, ymax, label, round_id])

                images_used.add(img_path)

    # Write CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["image_path", "xmin", "ymin", "xmax", "ymax", "label"]
        if round_id is not None:
            header.append("round")
        w.writerow(header)
        w.writerows(rows)

    return len(rows), len(images_used)


def export_project_tasks(access: str,
                         project_id: int,
                         finished_only: bool = True) -> List[Dict[str, Any]]:
    """Export tasks (with annotations) from a Label Studio project.

    If finished_only is True, requests the server to return only
    completed tasks.
    """
    url = (f"{BASE_URL}/api/projects/{project_id}/export"
           f"?exportType=JSON&download_all_tasks=true"
           f"{'&onlyFinished=1' if finished_only else ''}")
    r = requests.get(url, headers=Health_check(access), timeout=60)
    r.raise_for_status()
    data = r.json()
    if finished_only and isinstance(data, list):
        data = _filter_finished_only(data)
    return data


def find_image_field(task_data: Dict[str, Any]) -> Optional[str]:
    """Searches for an image file path in the provided task data dictionary.

    Iterates through the key-value pairs in `task_data` and returns the value of the first key
    whose value is a string containing a common image file extension (e.g., .jpg, .png, .tiff).

    Args:
        task_data (Dict[str, Any]): A dictionary containing task data, potentially including image file paths.

    Returns:
        Optional[str]: The image file path if found, otherwise None.
    """
    for k, v in task_data.items():
        if isinstance(v, str) and any(ext in v.lower(
        ) for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff"]):
            return v
    return None


def absolute_image_url(rel_or_abs: str) -> str:
    """Returns an absolute image URL.

    If the input string is already an absolute URL (starts with "http://" or "https://"),
    it is returned unchanged. Otherwise, the input is treated as a relative path and
    concatenated with the BASE_URL to form an absolute URL.

    Args:
        rel_or_abs (str): A relative or absolute image URL.

    Returns:
        str: An absolute image URL.
    """
    if rel_or_abs.startswith("http://") or rel_or_abs.startswith("https://"):
        return rel_or_abs
    return f"{BASE_URL.rstrip('/')}/{rel_or_abs.lstrip('/')}"


def parse_image_url(image_url: str) -> str:
    """Extracts the image filename from a given image URL.

    The function attempts to parse the filename from the URL by checking for a query parameter 'd'.
    If not found, it parses the URL path and query string to extract the filename.
    If extraction fails, it returns "unknown_image.jpg".

    Args:
        image_url (str): The URL of the image.

    Returns:
        str: The extracted image filename, or "unknown_image.jpg" if extraction fails.
    """
    try:
        if "?d=" in image_url:
            filename = image_url.split("?d=")[-1]
        else:
            parsed = urlparse(image_url)
            filename = pathlib.Path(parsed.path).name
            if parsed.query:
                qs = parse_qs(parsed.query)
                if "d" in qs:
                    filename = qs["d"][0]
        return pathlib.Path(filename.lstrip("/")).name or "unknown_image.jpg"
    except Exception:
        return "unknown_image.jpg"


def extract_annotation_pairs(task: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Extracts pairs of image filenames and annotation labels from a Label
    Studio task.

    This function processes a Label Studio task dictionary to extract annotation pairs,
    where each pair consists of the image filename and an associated label or annotation value.
    It supports multiple annotation types, including choices, textarea, and any field ending
    with "labels" (e.g., rectanglelabels, polygonlabels, etc.).

    Args:
        task (Dict[str, Any]): A dictionary representing a Label Studio task, expected to contain
            image data and annotation results.

    Returns:
        List[Tuple[str, str]]: A list of tuples, each containing the image filename and a label or
            annotation value extracted from the task.
    """
    pairs: List[Tuple[str, str]] = []
    image_field = find_image_field(task.get("data", {}) or {})
    if not image_field:
        return pairs
    filename = parse_image_url(image_field)

    anns = task.get("annotations") or []
    for ann in anns:
        for res in ann.get("result") or []:
            t = res.get("type")
            val = (res.get("value") or {})
            # choices
            if t == "choices":
                for c in val.get("choices", []):
                    pairs.append((filename, str(c)))
            # textarea
            if t == "textarea":
                texts = val.get("text", [])
                if texts:
                    pairs.append((filename, str(texts[0])))
            # any "*labels" list (rectanglelabels, polygonlabels, brushlabels, keypointlabels, labels, taxonomyLabels, etc.)
            for key, v in val.items():
                if key.lower().endswith("labels") and isinstance(v, list):
                    for lab in v:
                        pairs.append((filename, str(lab)))
    return pairs
