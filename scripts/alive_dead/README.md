# Alive/Dead CropModel training

Scripts that build the training data for `weecology/cropmodel-deadtrees`, make
leak-free splits, and train and evaluate the model.

The scripts read and write only inside one data root. The Weecology copy lives on
HiPerGator at `/blue/ewhite/b.weinstein/DeepForest/alive_dead`; given a copy of
`sources/`, every later step is reproducible.

```
<root>/
├── sources/      raw inputs, unchanged
├── crops/        crops derived from sources, one folder per source
├── manifests/    one CSV row per crop
├── splits/       grouped train/val/test splits as ImageFolder trees of symlinks
└── runs/         checkpoints, logs and evaluation results
```

## Sources

| Source | Labels | Used for |
|---|---|---|
| NEON hand-labelled crops (6,329 after removing duplicates and conflicts) | hand | fine-tuning and test |
| NEON Woody Vegetation Structure (VST) dead and pre-death trees, RELEASE-2026, matched to crowns in NEON RGB, 2,120 kept after a human cull | weak, reviewed | fine-tuning (train only) |
| [deadtrees.earth DTE-aerial-train](https://huggingface.co/datasets/ayushi3536/deadtree.earth-aerial-train_cc_by), CC BY, 0.1 m | crowns detected by DeepForest, labelled from pseudo-label masks | pretraining |
| [deadtrees.earth DTE-aerial-bench](https://huggingface.co/datasets/ayushi3536/deadtree.earth-aerial-bench), 10 cm | crowns labelled from expert masks | evaluation only |
| [TreeFinder](https://www.kaggle.com/datasets/zhihaow/tree-finder), NAIP 0.6 m | dead-blob boxes and detector crowns | tested for pretraining, not used |

## Steps

Each manifest row has `crop_path`, `label` (`Alive` or `Dead`), `source` and
`group`. The group is the spatial unit that must not be split: the 1 km NEON tile,
the deadtrees.earth orthophoto, or the source NAIP image.

```bash
ROOT=/blue/ewhite/b.weinstein/DeepForest/alive_dead

# 1. Manifests
python build_neon_hand_manifest.py --root $ROOT
python build_dte_crops.py --root $ROOT --source train    # GPU
python build_dte_crops.py --root $ROOT --source bench    # GPU

# 2. Grouped splits (no group, and no identical image, in more than one split)
python make_split.py --root $ROOT --manifest neon_hand.csv --name neon_hand_v1
python make_split.py --root $ROOT --manifest dte_train.csv --name dte_train_v2 --n-folds 10
python make_split.py --root $ROOT --manifest dte_bench.csv --name dte_bench_v2 --all-test

# 3. NEON VST crowns: survey tables, stem coordinates, RGB/CHM tiles, crops,
#    then a human review that deletes obvious mismatches
NEON_TOKEN=... python download_vst.py --out_dir $ROOT/sources/neon_vst
python merge_vst.py --vst_dir $ROOT/sources/neon_vst/RELEASE-2026 \
    --tos_dir $ROOT/sources/neon_tos_plots_V12/<unzipped folder>
sbatch fetch_vst_tiles.sbatch                  # reads the token from ~/.neon_token
sbatch build_vst_crops.sbatch                  # GPU; also writes the review zip
python apply_vst_review.py --root $ROOT --review-dir <returned review folder>
python make_split.py --root $ROOT --base-split neon_hand_v1 \
    --train-only neon_vst_reviewed.csv --name neon_hand_vst_v1

# 4. Pretrain on deadtrees.earth crowns, then fine-tune on the NEON crowns
sbatch train.sbatch dte_train_v1 pretrain_dte_train_v1_resnet50 --epochs 8 --num-workers 16
sbatch train.sbatch neon_hand_vst_v1 finetune_neon_hand_vst_v1_from_dte_train_v1 \
    --init-from $ROOT/runs/pretrain_dte_train_v1_resnet50/best.ckpt \
    --extra-eval $ROOT/splits/dte_bench_all $ROOT/splits/dte_bench_v2

# 5. Evaluate any checkpoint or Hugging Face model on a split
python evaluate_cropmodel.py --split-dir $ROOT/splits/dte_bench_v2 --split test \
    --model weecology/cropmodel-deadtrees --out $ROOT/runs/hf_current/eval_dte_bench_v2
```

`make_split.py` needs scikit-learn (installed with `uv sync --all-extras --dev`).

## Labelling rules for deadtrees.earth crowns

DeepForest (`weecology/deepforest-tree`, score >= 0.3) detects crowns on each
1024 px patch. A crown is Dead when at least 50% of its box is mortality, and
Alive when it has no mortality pixel and at least 70% tree cover. Crowns in
between, or touching ignored pixels, are skipped. Each crop is padded by 10 px
(1 m).

Crowns that are likely partial trees are then removed. The unfiltered manifest
is kept as `dte_<source>_unfiltered.csv`, and `--refilter` re-applies the
filters without running detection again.

| Filter | Default | Removed from DTE train |
|---|---|---|
| crown smaller than `--min-side-m` (square root of box area) | 2 m | 15% |
| box aspect ratio above `--max-aspect` | 2 | 8%, mostly crowns cut by the patch edge |
| more than `--max-nested` of the box inside a larger crown box | 0.5 | 2% |
| crown box touching the patch edge | on | only for manifests that record `crown_*` boxes |

Size filtering doesn't fix the main remaining issue: the detector often splits a
large crown into several 2–4 m boxes, and those fragments are kept.

## NEON VST crowns

`build_vst_crops.py` takes every tree ever recorded as standing dead or dead with a
broken bole, at least 3 m tall and with a mapped stem, and pairs it with every
RGB flight over its plot. A flight is labelled Dead when the last field visit on
or before it recorded the tree dead and no later visit recorded it live, and
Alive for flights before the tree died. The nearest detected crown is cropped,
and crowns whose lidar canopy height disagrees with the field height are dropped.
Most dead stems are under live canopy, so a person then deleted 81% of the Dead
candidates (`apply_vst_review.py`). Tiles in the `neon_hand_v1` validation and
test splits are excluded, and `make_split.py --base-split` keeps those splits
unchanged.

## Known issues in the NEON hand labels

- The original `dead_train`/`dead_test` split shared 78 of 95 source tiles.
  `neon_hand_v1` replaces it with a tile-grouped split.
- 47 images were labelled both Alive and Dead and are dropped
  (`manifests/neon_hand_conflicts.csv`).
- The crops have no saved source coordinates, so they can't be re-cut. Dead crops
  are larger than Alive crops (median 85 vs 53 px).
