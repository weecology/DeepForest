# Overview

Active learning in the context of image object detection is a technique for efficiently selecting the most valuable images and objects to annotate, reducing the labeling effort required for model training while maximizing performance gains. This submodule provides active learning utilities. It wraps DeepForest’s training and inference APIs with a small, reproducible loop that selects informative unlabeled images using an entropy-based acquisition function. It is designed to be extensible, so you can swap selection strategies, wire it to an annotation tool (Label Studio), and run multi-round training.

# Key Features

* Reproducible training rounds with seeds and configurable hardware
* Minimal configuration via a `Config` dataclass
* Lightning-based training with checkpointing and early stopping
* Batch prediction over image lists
* Entropy-based acquisition that scores each image by uncertainty
* CSV- and image-path–based I/O fully compatible with DeepForest
* Ready-to-wire hooks for Label Studio integration (pre-annotations, export/import)

# Directory Layout

* `workdir/`

  * `logs/` Lightning logs and checkpoints
  * `acquisition/` acquisition manifests (`selection_round.csv`)
* `images_dir/` all image files accessible by relative `image_path`
* `train.csv`, `val.csv` DeepForest-format CSVs with columns: `image_path,xmin,ymin,xmax,ymax,label`

# Quickstart

```python
from active_learning import Config, ActiveLearner

cfg = Config(
    workdir="/tmp/df_active",
    images_dir="/data/trees/images",
    train_csv="/data/trees/train.csv",
    val_csv="/data/trees/val.csv",
    classes=["Tree"],
    epochs_per_round=10,
    batch_size=4,
    precision=32,
    device="auto",
    k_per_round=50,
)

al = ActiveLearner(cfg)
ckpt = al.fit_one_round()                 # train for one round
metrics = al.evaluate()                   # evaluate on val set

# Select 50 uncertain images from a list or a file of paths
manifest = al.select_for_labeling("/data/trees/unlabeled_paths.txt")
print(manifest.head())
```

# How It Works

1. **Training**: `ActiveLearner.fit_one_round()` creates a PL Trainer with checkpointing and early stopping, then trains a DeepForest model for `epochs_per_round`.
2. **Evaluation**: `ActiveLearner.evaluate()` runs DeepForest’s evaluation over `val_csv` with the configured IoU threshold.
3. **Prediction**: `ActiveLearner.predict_images(paths)` runs `model.predict_image` on each path and returns a dict of DataFrames.
4. **Acquisition**: `select_for_labeling()` aggregates per-image predictions into a class-score distribution, computes Shannon entropy, and returns a ranked DataFrame (highest entropy first). It also saves `acquisition/selection_round.csv` for traceability.

# Configuration Reference

`Config` dataclass fields and effects:

* **workdir**: Root for logs, checkpoints, acquisition manifests
* **images\_dir**: Base directory for all images referenced by CSVs
* **train\_csv, val\_csv**: DeepForest-format CSVs
* **classes**: List of class labels; sets `model.config["num_classes"]`
* **epochs\_per\_round**: Max epochs per training round
* **batch\_size**: Sets `model.config["batch_size"]`
* **lr, weight\_decay**: Optimizer hyperparameters (reserved for future use if you extend the Trainer/model init)
* **precision**: PL precision (e.g., 16/32/"bf16")
* **device**: `"auto"`, `"cpu"`, or `"cuda:0"` etc. Resolved to PL `accelerator`/`devices`
* **num\_workers**: Dataloader workers (if wired into DataModules in extensions)
* **seed**: Seeds Python/NumPy/Torch and calls `pl.seed_everything`
* **use\_release\_weights**: If `True`, calls `model.use_release()` to warm start
* **iou\_eval**: IoU threshold used by `evaluate()`
* **k\_per\_round**: Default number of images to acquire
* **score\_threshold\_pred**: Score threshold for predictions during acquisition

# API Reference

## Utility Functions

* `_seed_everything(seed)`

  * Seeds Python, NumPy, Torch; tries `pl.seed_everything(..., workers=True)`.

* `_resolve_device(device)`

  * Returns `(accelerator, devices)` for PL Trainer. Supports "auto", explicit CUDA, and CPU fallback.

* `_ensure_dir(pathlike)`

  * Creates the directory path if it doesn’t exist.

* `_read_paths_file(path_or_list)`

  * Accepts a list/tuple of paths, or a text file containing newline-separated paths. Returns a list of string paths.

* `_image_entropy_from_predictions(pred_df, classes)`

  * Aggregates detection `score` per `label` and computes Shannon entropy over the normalized class distribution. Empty predictions are treated as maximally uncertain (`log(C)`). Returns `(entropy, n_preds, mean_score)`.

## Class: ActiveLearner

Constructor

```python
ActiveLearner(cfg: Config)
```

* Creates work, logs, and acquisition directories
* Initializes DeepForest model with correct class count and batch size
* Optionally loads release weights
* Builds a PL Trainer with ModelCheckpoint and EarlyStopping
* Attaches train/val CSVs and roots to `model.config`

Methods

* `fit_one_round() -> Path`

  * Trains for `epochs_per_round`. Returns best checkpoint path.

* `evaluate() -> dict`

  * Runs validation evaluation via DeepForest’s API. Returns a dict with metrics and optional DataFrames.

* `predict_images(paths) -> dict[str, pd.DataFrame]`

  * Predicts on each image path. Returns mapping `path -> predictions_df`. Missing or erroring images yield empty DataFrames with proper columns.

* `select_for_labeling(unlabeled_paths, k=None) -> pd.DataFrame`

  * Computes entropy per image from predictions, ranks descending, writes `acquisition/selection_round.csv`, and returns the top `k` rows with columns: `image_path, entropy, n_preds, mean_score`.

# Input & Output Formats

## DeepForest CSV (training/validation)

Columns required: `image_path,xmin,ymin,xmax,ymax,label`

* `image_path` can be absolute or relative to `images_dir`
* Coordinates are pixel units in the image coordinate system

## Prediction DataFrame

Returned by `model.predict_image` and normalized here to columns:

* `xmin, ymin, xmax, ymax, label, score, image_path`

# Logging, Checkpoints, and Reproducibility

* Checkpoints saved under `workdir/logs/checkpoints/`
* Best model path tracked via `ModelCheckpoint`
* Training is deterministic where possible; seeds are set for Python, NumPy, and Torch
* Early stopping monitors `val_map` with patience 3 by default

# Error Handling Notes

* Prediction exceptions are caught per-image; an empty DataFrame is substituted
* Evaluation failure returns an empty dict and logs a warning
* CUDA unavailability falls back to CPU with a warning if CUDA was explicitly requested

# Example: Multi-Round Active Learning Loop

```python
al = ActiveLearner(cfg)
for round_id in range(5):
    print(f"Round {round_id}")
    ckpt = al.fit_one_round()
    print(al.evaluate())
    manifest = al.select_for_labeling("/data/unlabeled_paths.txt", k=cfg.k_per_round)
    # Send `manifest` to an annotation workflow (e.g., Label Studio)
    # Merge newly labeled data into train.csv, deduplicate, and continue
```

# Label Studio Integration Plan

This section outlines how to connect the acquisition outputs to Label Studio for annotation and then flow the results back into DeepForest.

## Workflow

1. **Select images** with `select_for_labeling()` to produce `selection_round.csv`.
2. **Upload images** to Label Studio using the SDK or UI.
3. **Create a project** with a bounding-box labeling config and class set derived from `cfg.classes`.
4. **(Optional) Pre-annotate** by pushing model predictions as prelabels to speed up annotation.
5. **Annotate** in Label Studio.
6. **Export annotations** as JSON.
7. **Convert** Label Studio JSON to DeepForest CSV and append to `train_csv` for the next round.

## Labeling Config Template (Bounding Boxes)

Replace the `choices` with your classes.

```xml
<View>
  <Image name="img" value="$image" zoom="true" rotateControl="true"/>
  <RectangleLabels name="label" toName="img" showInline="true">
    <Label value="Tree" background="#87CEFA"/>
    <!-- add more classes as needed -->
  </RectangleLabels>
</View>
```

## Creating a Project and Uploading Tasks

```python
from label_studio_sdk import Client

ls = Client(url="http://localhost:8080", api_key="YOUR_API_KEY")
project = ls.start_project(
    title="DeepForest Active Learning",
    label_config=open("bbox_config.xml").read(),
    description="AL round annotations"
)

# Upload image paths selected for this round
import pandas as pd
sel = pd.read_csv("/tmp/df_active/acquisition/selection_round.csv")

# Each task data can point to a local file path or a URL accessible by the server
tasks = [{"data": {"image": path}} for path in sel.image_path.tolist()]
project.import_tasks(tasks)
```

## Pushing Pre-Annotations (Optional)

Use your current model to create prelabels so annotators edit rather than draw from scratch. Convert pixel boxes to Label Studio relative percentages.

```python
import json
from PIL import Image

results = []
for img_path, df in al.predict_images(sel.image_path.tolist()).items():
    w, h = Image.open(img_path).size
    for _, r in df.iterrows():
        x = 100 * r["xmin"] / w
        y = 100 * r["ymin"] / h
        width = 100 * (r["xmax"] - r["xmin"]) / w
        height = 100 * (r["ymax"] - r["ymin"]) / h
        results.append({
            "data": {"image": img_path},
            "predictions": [{
                "result": [{
                    "from_name": "label",
                    "to_name": "img",
                    "type": "rectanglelabels",
                    "value": {
                        "x": x, "y": y, "width": width, "height": height,
                        "rotation": 0, "rectanglelabels": [str(r["label"])]
                    },
                    "score": float(r.get("score", 0.0))
                }]
            }]
        })

# Bulk import with predictions
project.import_tasks(results)
```

## Converting Label Studio Export to DeepForest CSV

Label Studio’s JSON export stores boxes in percentages; convert them back to pixels using the source image size.

```python
import json
import pandas as pd
from PIL import Image
from pathlib import Path

def labelstudio_to_deepforest(ls_export_json: str, out_csv: str):
    rows = []
    data = json.load(open(ls_export_json, "r", encoding="utf-8"))
    for task in data:
        img_path = task["data"]["image"]
        # Resolve to filesystem path if needed
        p = Path(img_path)
        if not p.exists():
            # Add your own mapping logic here (e.g., strip URL prefix)
            continue
        w, h = Image.open(p).size
        for ann in task.get("annotations", []):
            for res in ann.get("result", []):
                if res.get("type") != "rectanglelabels":
                    continue
                v = res["value"]
                xmin = v["x"] * w / 100.0
                ymin = v["y"] * h / 100.0
                xmax = xmin + v["width"] * w / 100.0
                ymax = ymin + v["height"] * h / 100.0
                label = v["rectanglelabels"][0]
                rows.append({
                    "image_path": str(p),
                    "xmin": int(round(xmin)),
                    "ymin": int(round(ymin)),
                    "xmax": int(round(xmax)),
                    "ymax": int(round(ymax)),
                    "label": label,
                })
    pd.DataFrame(rows).to_csv(out_csv, index=False)

# Example
# labelstudio_to_deepforest("/exports/project-1-at-2025-08-25-10-00-00.json",
#                           "/data/trees/new_round_labels.csv")
```

## Automating the Round-Trip

* After annotators finish a batch, export JSON, convert to DeepForest CSV, and append to `train_csv`.
* Deduplicate by `(image_path, xmin, ymin, xmax, ymax, label)` if necessary.
* Optionally use Label Studio webhooks to trigger a small script that runs the conversion and kicks off the next `fit_one_round()`.

# Extending the Acquisition Strategy

The current entropy score uses class-distribution uncertainty from detection scores. You can plug in other criteria:

* **Score margin**: difference between top-2 class masses
* **Mean score**: prioritize low-confidence images
* **Diversity**: add image embeddings and do k-center or clustering over features
* **Spatial entropy**: weight by number of boxes or box-area variance
* **Cost-aware**: penalize large, hard-to-annotate images
* **BALD/MC-Dropout**: approximate Bayesian uncertainty via stochastic forward passes

Tip: return a composite score and store components in the manifest for auditability.

# Submodule Placement in DeepForest

Recommended layout inside the DeepForest repo:

* `deepforest/active_learning/`

  * `__init__.py` exports `Config`, `ActiveLearner`
  * `acquire.py` selection utilities (entropy, margin, diversity)
  * `label_studio.py` import/export helpers and SDK wiring
  * `cli.py` small CLI for common workflows
  * `README.md` quickstart plus examples

Packaging notes

* Keep Label Studio as an **optional extra**: `pip install deepforest[active]` or `pip install label-studio-sdk`
* Avoid changing `deepforest.main.deepforest` internals; interact via its public API

# CLI Sketch

```bash
# Train one round
python -m deepforest.active_learning.cli \
  --workdir /tmp/df_active \
  --images_dir /data/trees/images \
  --train_csv /data/trees/train.csv \
  --val_csv /data/trees/val.csv \
  --classes Tree \
  fit

# Acquire top-K unlabeled images from a text file
python -m deepforest.active_learning.cli acquire \
  --unlabeled /data/unlabeled.txt \
  --k 50

# Convert Label Studio export to DeepForest CSV
python -m deepforest.active_learning.cli ls2csv \
  --export /exports/project.json \
  --out /data/new_labels.csv
```

