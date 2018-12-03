#!/bin/bash
stamp=$(date +%Y%m%d_%H%M%S) # time stamp for a directory to catch all logs.

sbatch --job-name=$stamp.run --output=$stamp.out --export=stamp=$stamp /home/b.weinstein/DeepLidar/HPC/training.sbatch
