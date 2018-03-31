#! /usr/bin/env sh
git clone https://github.com/LASzip/LASzip.git
cd LASzip
git checkout tags/2.0.2

mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/LASzip/build 
make
make install

cd ~
unzip tests/LAStools.zip
cd LAStools
make
cd bin

cp laszip ~/LASzip/build/bin/
cd ~/LASzip/build/bin/
ln -s laszip laszip-cli

export LD_LIBRARY_PATH="~/LASzip/build/lib:$LD_LIBRARY_PATH"
export PATH="~/LASzip/build/bin:$PATH"