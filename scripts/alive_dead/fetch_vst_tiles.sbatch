#!/bin/bash
#SBATCH --job-name=vst_tiles
#SBATCH --account=ewhite
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --output=/home/b.weinstein/logs/vst_tiles_%j.out
#SBATCH --error=/home/b.weinstein/logs/vst_tiles_%j.err

# Download NEON RGB/CHM tiles covering VST dead-tree candidates. Restartable: finished tiles are skipped.
# Reads the NEON API token from ~/.neon_token.
set -euo pipefail
SRC=/blue/ewhite/b.weinstein/DeepForest/alive_dead/sources
REPO=/blue/ewhite/b.weinstein/src/DeepForest-alive-dead
export NEON_TOKEN=$(tr -d '[:space:]' < ~/.neon_token)
$REPO/.venv/bin/python -I $REPO/scripts/alive_dead/fetch_vst_tiles.py \
    --vst_dir $SRC/neon_vst/RELEASE-2026 --out_dir $SRC/neon_aop --workers 8 "$@"
