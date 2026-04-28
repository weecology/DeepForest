# Marine Mammal Detector Reproducibility

This folder provides a concise, reproducible workflow to train a general marine
mammal detector with DeepForest from BOEM/USGS-style detection CSVs on HPC.

## 1) Prepare marine-mammal-only CSVs

Input CSVs should already exist on HPC (for example outputs from a BOEM
preparation workflow).

```bash
uv run python reproducibility/marine_mammal_detector/prepare_marine_mammal_data.py \
  --train-csv "/path/to/crops/train.csv" \
  --test-csv "/path/to/crops/test.csv" \
  --zero-shot-csv "/path/to/crops/zero_shot.csv" \
  --output-dir "/path/to/crops/marine_mammal"
```

By default this keeps rows whose labels contain marine mammal keywords
(`seal`, `dolphin`, `whale`, etc.) and keeps empty-image rows. Labels are
normalized to a single class, `Object`.

## 2) Train on HPC

Edit paths in `train_marine_mammal_detector.slurm`, then submit:

```bash
sbatch reproducibility/marine_mammal_detector/train_marine_mammal_detector.slurm
```

Training script:

```bash
uv run python reproducibility/marine_mammal_detector/train_marine_mammal_detector.py \
  --train-csv "/path/to/crops/marine_mammal/train.csv" \
  --test-csv "/path/to/crops/marine_mammal/test.csv" \
  --zero-shot-csv "/path/to/crops/marine_mammal/zero_shot.csv" \
  --root-dir "/path/to/crops" \
  --log-root "/path/to/logs" \
  --experiment-name "marine-mammal-v1" \
  --strategy "ddp" \
  --tensorboard
```

Optional Hugging Face upload:

```bash
HF_TOKEN=... uv run python reproducibility/marine_mammal_detector/train_marine_mammal_detector.py \
  ... \
  --hub-repo "your-user-or-org/deepforest-marine-mammal"
```

This follows the contribution workflow in `CONTRIBUTING.md`:
- Ensure `label_dict = {"Object": 0}`
- Push with `push_to_hub`
- Open a DeepForest PR linking the model repo and dataset provenance

## 3) Make blog-ready visuals

```bash
uv run python reproducibility/marine_mammal_detector/visualize_marine_mammal_results.py \
  --annotations-csv "/path/to/crops/marine_mammal/test.csv" \
  --root-dir "/path/to/crops" \
  --checkpoint "/path/to/logs/marine-mammal-v1/<timestamp>/checkpoints/last.ckpt" \
  --output-dir "/path/to/blog_figures" \
  --num-images 8
```

Outputs per sampled image:
- Ground-truth bounding box figure
- Prediction figure from the trained model

## Notes on drone-image generalization

To assess transfer from lower-resolution BOEM imagery to higher-resolution drone
imagery, evaluate the same checkpoint on a held-out drone CSV (same schema) and
compare precision/recall and qualitative overlays against BOEM test/zero-shot.
