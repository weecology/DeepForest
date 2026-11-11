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
python make_split.py --root $ROOT --manifest dte_train.csv --name dte_train_v1 --n-folds 10
python make_split.py --root $ROOT --manifest dte_bench.csv --name dte_bench_all --all-test

# 3. Pretrain on deadtrees.earth crowns, then fine-tune on the hand labels
sbatch train.sbatch dte_train_v1 pretrain_dte --epochs 8 --num-workers 16
sbatch train.sbatch neon_hand_v1 finetune_dte \
    --init-from $ROOT/runs/pretrain_dte/best.ckpt --extra-eval $ROOT/splits/dte_bench_all

# 4. Evaluate any checkpoint or Hugging Face model on a split
python evaluate_cropmodel.py --split-dir $ROOT/splits/dte_bench_all --split test \
    --model weecology/cropmodel-deadtrees --out $ROOT/runs/hf_current/eval_dte_bench_all
```

`make_split.py` needs scikit-learn (installed with `uv sync --all-extras --dev`).

## Labelling rules for deadtrees.earth crowns

DeepForest (`weecology/deepforest-tree`, score >= 0.3) detects crowns on each
1024 px patch. A crown is Dead when at least 50% of its box is mortality, and
Alive when it has no mortality pixel and at least 70% tree cover. Crowns in
between, or touching ignored pixels, are skipped. Each crop is padded by 10 px
(1 m).

## Known issues in the NEON hand labels

- The original `dead_train`/`dead_test` split shared 78 of 95 source tiles.
  `neon_hand_v1` replaces it with a tile-grouped split.
- 47 images were labelled both Alive and Dead and are dropped
  (`manifests/neon_hand_conflicts.csv`).
- The crops have no saved source coordinates, so they can't be re-cut. Dead crops
  are larger than Alive crops (median 85 vs 53 px).
