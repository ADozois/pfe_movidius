#!/bin/bash

# Mise à jour su système
sudo apt-get update
sudo apt-get upgrade -y

# Téléchargement des dépendances
sudo apt-get install -y libusb-1.0-0-dev libprotobuf-dev
sudo apt-get install -y libleveldb-dev libsnappy-dev
sudo apt-get install -y libopencv-dev
sudo apt-get install -y libhdf5-serial-dev protobuf-compiler
sudo apt-get install -y libatlas-base-dev git automake 
sudo apt-get install -y byacc lsb-release cmake 
sudo apt-get install -y libgflags-dev libgoogle-glog-dev 
sudo apt-get install -y liblmdb-dev swig3.0 graphviz 
sudo apt-get install -y libxslt-dev libxml2-dev 
sudo apt-get install -y gfortran 
sudo apt-get install -y python3-dev python-pip python3-pip 
sudo apt-get install -y python3-setuptools python3-markdown 
sudo apt-get install -y python3-pillow python3-yaml python3-pygraphviz
sudo apt-get install -y python3-h5py python3-nose python3-lxml 
sudo apt-get install -y python3-matplotlib python3-numpy 
sudo apt-get install -y python3-protobuf python3-dateutil 
sudo apt-get install -y python3-skimage python3-scipy 
sudo apt-get install -y python3-six python3-networkx
sudo apt-get install -y python-opencv

# Téléchargement du code de Movidius
mkdir -p ~/workspace
cd ~/workspace
git clone -b ncsdk2 https://github.com/movidius/ncsdk.git
cd ~/workspace/ncsdk/api/src

# Compilation et installation du code
make
sudo make install

cd ..
git clone https://github.com/ADozois/pfe_movidius.git
