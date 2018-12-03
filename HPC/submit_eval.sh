#!/bin/bash
stamp=$(date +%Y%m%d_%H%M%S) # time stamp for a directory to catch all logs.
sbatch --job-name=$stamp.run --output=/home/b.weinstein/logs/$stamp.out --export=stamp=/home/b.weinstein/logs/$stamp /home/b.weinstein/DeepLidar/HPC/eval.sbatch
