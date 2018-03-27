#!/bin/bash

#Push code to dev branch
git push

#ssh into hipergator
#TODO update ssh keychain?
ssh  b.weinstein@hpg2.rc.ufl.edu

#ENTER PASSWORD

#Clone latest data
cd path/to/repo

git pull

#Checkout new branch
git branch date

####Lidar
sbatch SubmitLidar.sh

### Submit Training Tree Crown Segmentation
###Hyperspectral and RGB segmentation
sbatch SubmitKeras.sh

###Evalutation Tree Crown Segmentation
sbatch Evaluation.sh

#Create results visualization
sbatch SubmitRVis.sh

#TODO wait until job ids are complete?

#Push results to github
#git push repo
