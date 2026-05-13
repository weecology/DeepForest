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

## 2) Add NOAA Arctic Seals RGB data

The [NOAA Arctic Seals 2019](https://lila.science/datasets/noaa-arctic-seals-2019/)
LILA dataset includes 14,311 RGB seal boxes across 4,113 annotated RGB images.
Use the setup script to sample a small image-level subset and write download
manifests; it does not download images.

```bash
uv run python reproducibility/marine_mammal_detector/prepare_noaa_arctic_seals.py \
  --output-dir "/path/to/crops/marine_mammal/noaa_arctic_seals" \
  --max-images 500 \
  --boem-train-csv "/path/to/crops/marine_mammal/train.csv" \
  --boem-test-csv "/path/to/crops/marine_mammal/test.csv"
```

Outputs include:
- `noaa_arctic_seals_train.csv` and `noaa_arctic_seals_test.csv`
- `train_with_noaa_arctic_seals.csv` and `test_with_noaa_arctic_seals.csv` when
  BOEM train/test CSVs are provided
- `noaa_arctic_seals_rgb_manifest.csv`, plus GCP/AWS URI lists for later HPC
  transfer

The generated CSVs use paths like `noaa-kotz/Images/...`. Download RGB images
under that folder inside the training `root_dir` before training with the
combined CSVs. Use `--max-images 0` to prepare the full annotated RGB set.

## 3) Prepare Zenodo zero-shot datasets

Two UAV cetacean datasets are useful for out-of-domain zero-shot checks:
- Bigal et al. 2022 dolphin images: <https://zenodo.org/records/7013418>
- Gray et al. 2019 cetacean images and VIA labels:
  <https://zenodo.org/records/4933008>

First, write Zenodo file manifests and download scripts. This does not download
the ZIP files.

```bash
uv run python reproducibility/marine_mammal_detector/prepare_zenodo_zero_shot.py \
  manifest \
  --dataset all \
  --output-dir "/path/to/crops/marine_mammal/zenodo_zero_shot"
```

After downloading and extracting on HPC, convert Gray et al. VIA polygons to
DeepForest boxes:

```bash
uv run python reproducibility/marine_mammal_detector/prepare_zenodo_zero_shot.py \
  gray-cetaceans \
  --extract-root "/path/to/extracted/gray_cetaceans" \
  --output-csv "/path/to/crops/marine_mammal/zero_shot/gray_cetaceans.csv"
```

Bigal et al. is organized by species and sea state but does not include box
annotations in the Zenodo record. For qualitative checks only, create whole-image
proxy boxes:

```bash
uv run python reproducibility/marine_mammal_detector/prepare_zenodo_zero_shot.py \
  bigal-dolphins \
  --extract-root "/path/to/extracted/bigal_dolphins" \
  --output-csv "/path/to/crops/marine_mammal/zero_shot/bigal_dolphins_whole_image.csv" \
  --manifest-csv "/path/to/crops/marine_mammal/zero_shot/bigal_dolphins_images.csv"
```

Use Gray et al. for quantitative zero-shot detection metrics. Treat Bigal et al.
as a qualitative drone-domain visual check unless bounding boxes are added.

## 4) Train on HPC

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

## 5) Make blog-ready visuals

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
