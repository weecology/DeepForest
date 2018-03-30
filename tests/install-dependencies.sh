#! /usr/bin/env sh
module load git
git clone https://github.com/LASzip/LASzip.git
cd LASzip
git checkout tags/2.0.2

mkdir build
cd build
module load cmake
module load gcc
cmake .. 
make
make install

cd ~
wget http://lastools.org/download/LAStools.zip
unzip LAStools.zip
cd LAStools
make

cd /usr/local/LAStools/bin
cp laszip /user/b.weinstein/LASzip/build/bin/
cd /usr/local/LASzip/build/bin/
ln -s /usr/local/LASzip/build/bin/laszip /usr/local/LASzip/build/bin/laszip-cli