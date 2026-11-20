#!/bin/bash
#SBATCH --job-name=vst_crops
#SBATCH --account=ewhite
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --output=/home/b.weinstein/logs/vst_crops_%j.out
#SBATCH --error=/home/b.weinstein/logs/vst_crops_%j.err

# Build weak Alive/Dead crops from NEON VST stems. Usage:
#   sbatch build_vst_crops.sbatch [extra build_vst_crops.py args, e.g. --limit 5]
set -euo pipefail
REPO=/blue/ewhite/b.weinstein/src/DeepForest-alive-dead
cd $REPO/scripts/alive_dead
$REPO/.venv/bin/python build_vst_crops.py --root /blue/ewhite/b.weinstein/DeepForest/alive_dead \
    --score-model weecology/cropmodel-deadtrees "$@"

# Review images and zip for the human cull (skipped for --limit test runs).
if [[ " $* " != *" --limit "* ]]; then
    $REPO/.venv/bin/python make_vst_review.py --root /blue/ewhite/b.weinstein/DeepForest/alive_dead --workers 8
fi
