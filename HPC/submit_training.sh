#!/bin/bash
stamp=$(date +%Y%m%d_%H%M%S) # time stamp for a directory to catch all logs.

sbatch --job-name=$stamp.run --output=/home/b.weinstein/$stamp.out --export=stamp=/home/b.weinstein/$stamp /home/b.weinstein/DeepLidar/HPC/training.sbatch
