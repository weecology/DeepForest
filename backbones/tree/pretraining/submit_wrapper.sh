#!/bin/bash
for i in {1..150}
do
   sbatch submit_pretraining_loop.sh
done