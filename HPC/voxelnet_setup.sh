conda create -n voxelnet python=3.6
source activate voxelnet
conda install numpy
conda install numba shapely tensorflow scipy
conda install -c menpo opencv
conda install -c anaconda cython
conda install Pillow
conda install pandas

#edited the setup.py to capture numpy array
python3 setup.py build_ext --inplace

cd kitti_eval

#load boost
ml boost

#find the boost dir
echo $HPC_BOOST_DIR

#build evaluation metrics
g++ -o evaluate_object_3d_offline evaluate_object_3d_offline.cpp -I /apps/boost/1.59.0

#grant permissions to launch tests
chmod +x launch_test.sh
