#!/bin/bash
#SBATCH --job-name=NEON_SubmitKeras    # Job name
#SBATCH --mail-type=END               # Mail events (NONE, BEGIN, END, FAIL, AL$
#SBATCH --mail-user=ben.weinstein@weecology.org   # Where to send mail
#SBATCH --account=ewhite
#SBATCH --qos=ewhite-b

#SBATCH --ntasks=1                 # Number of MPI ranks
#SBATCH --cpus-per-task=1            # Number of cores per MPI rank
#SBATCH --time=00:59:00       #Time limit hrs:min:sec
#SBATCH --output=itc.out   # Standard output and error log
#SBATCH --error=itc.err
#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem-per-cpu=2000
#SBATCH --partition=hpg1-gpu
#SBATCH --gres=gpu:tesla:2

date;hostname;pwd

module load cuda/8.0

cudaMemTest=/ufrc/ufhpc/chasman/Cuda/cudaMemTest/cuda_memtest

cudaDevs=$(echo $CUDA_VISIBLE_DEVICES | sed -e 's/,/ /g')

for cudaDev in $cudaDevs
do
    echo cudaDev = $cudaDev
    #srun --gres=gpu:tesla:1 -n 1 --exclusive ./gpuMemTest.sh > gpuMemTest.out.$cudaDev 2>&1 &
    $cudaMemTest --num_passes 1 --device $cudaDev > gpuMemTest.out.$cudaDev 2>&1 &
done
wait

date
