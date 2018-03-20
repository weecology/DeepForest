/#!/usr/bin/env bash

#Setup script for interactive juypter notebook on HPC
#https://github.com/pangeo-data/pangeo/wiki/Getting-Started-with-Dask-on-Cheyenne

#Start interactive session
module load ufrc
srundev --time=04:00:00

#Isolate Env
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh

#home directory
cd ~
conda create -n pangeo -c conda-forge python=3.6 dask distributed xarray jupyterlab mpi4py
source activate pangeo
