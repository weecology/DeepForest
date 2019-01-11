#!/bin/bash
#SBATCH --job-name=DeepForest   # Job name
#SBATCH --mail-type=END               # Mail events (NONE, BEGIN, END, FAIL, AL$
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ranks
#SBATCH --ntasks=1                 # Number of MPI ranks
#SBATCH --cpus-per-task=5            # Number of cores per MPI rank
#SBATCH --mem-per-cpu=10GB
#SBATCH --time=24:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/DeepForest.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/DeepForest.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1

#export  HDF5_USE_FILE_LOCKING=FALSE

ml git
ml geos/3.6.2
ml tensorflow/1.10.1
export PYTHONPATH=${PYTHONPATH}:/home/b.weinstein/miniconda3/envs/DeepForest/lib/python3.6/site-packages/
echo $PYTHONPATH

cd /home/b.weinstein/DeepForest

python predict.py --model /orange/ewhite/b.weinstein/retinanet/snapshots/20181022_120303/resnet50_14.h5 --image /orange/ewhite/NeonData/SJER/DP3.30010.001/2018/FullSite/D17/2018_SJER_3/L3/Camera/Mosaic/V01/2018_SJER_3_255000_4106000_image.tif
