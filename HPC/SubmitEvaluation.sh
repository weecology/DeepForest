#!/bin/bash
#SBATCH --job-name=NEON_Evaluation    # Job name
#SBATCH --mail-type=END               # Mail events (NONE, BEGIN, END, FAIL, AL$
#SBATCH --mail-user=ben.weinstein@weecology.org   # Where to send mail
#SBATCH --account=ewhite
#SBATCH --qos=ewhite-b

#SBATCH --ntasks=1                 # Number of MPI ranks
#SBATCH --cpus-per-task=1            # Number of cores per MPI rank
#SBATCH --mem=4000
#SBATCH --time=00:59:00       #Time limit hrs:min:sec
#SBATCH --output=evaluation.out   # Standard output and error log
#SBATCH --error=evaluation.err

module load R

Rscript Evaluation.R