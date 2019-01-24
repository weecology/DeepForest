#!/bin/bash
stamp=$(date +%Y%m%d_%H%M%S) # time stamp for a directory to catch all logs.
sbatch --job-name=$stamp.run --output=/home/b.weinstein/logs/$stamp.out --error=/home/b.weinstein/logs/$stamp.err --export=stamp=/home/b.weinstein/logs/$stamp /home/b.weinstein/DeepLidar/HPC/retraining.sbatch
