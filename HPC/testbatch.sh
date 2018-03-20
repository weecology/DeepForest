#!/bin/bash
#SBATCH --job-name=NEON_LIDAR   # Job name
#SBATCH --mail-type=END               # Mail events (NONE, BEGIN, END, FAIL, AL$
#SBATCH --mail-user=ben.weinstein@weecology.org   # Where to send mail
#SBATCH --account=ewhite
#SBATCH --qos=ewhite-b

#SBATCH --ntasks=1                 # Number of MPI ranks
#SBATCH --cpus-per-task=1            # Number of cores per MPI rank
#SBATCH --mem=4000
#SBATCH --time=00:59:00       #Time limit hrs:min:sec
#SBATCH --output=NEON_LIDAR.out   # Standard output and error log
#SBATCH --error=NEON_LIDAR.err

module load R

Rscript session-info.R