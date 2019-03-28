#!/bin/bash
# install dependencies for gunpowder
# conda activate tf_cu90
# sudo apt-get update && sudo apt-get install -y --no-install-recommends \
#         build-essential \
#         cmake \
#         git \
#         libboost-all-dev \
#         python-dev \
#         python-numpy \
#         python-pip \
#         python-setuptools \
#         python-scipy && \
#     rm -rf /var/lib/apt/lists/*

pip install cython

MALIS_ROOT=malis_root
MALIS_REPOSITORY=https://github.com/TuragaLab/malis.git
MALIS_REVISION=2206fe01bd2d10c3bc6a861897820731d1ae131b
mkdir ${MALIS_ROOT}
git clone ${MALIS_REPOSITORY} ${MALIS_ROOT}
cd ${MALIS_ROOT}
git checkout ${MALIS_REVISION}
python setup.py build_ext --inplace
cd ..
export PYTHONPATH=${PWD}/${MALIS_ROOT}:$PYTHONPATH

AUGMENT_ROOT=augment_root
AUGMENT_REPOSITORY=https://github.com/funkey/augment.git
AUGMENT_REVISION=4a42b01ccad7607b47a1096e904220729dbcb80a 
mkdir ${AUGMENT_ROOT} 
git clone ${AUGMENT_REPOSITORY} ${AUGMENT_ROOT}
cd ${AUGMENT_ROOT} 
git checkout ${AUGMENT_REVISION}
pip install -r requirements.txt
cd ..
export PYTHONPATH=${PWD}/${AUGMENT_ROOT}:$PYTHONPATH

DVISION_ROOT=dvision
DVISION_REPOSITORY=https://github.com/TuragaLab/dvision.git
DVISION_REVISION=v0.1.1
mkdir ${DVISION_ROOT}
git clone -b ${DVISION_REVISION} --depth 1 ${DVISION_REPOSITORY} ${DIVISION_ROOT}
cd ${DVISION_ROOT}
pip install -r requirements.txt
cd ..
export PYTHONPATH=${PWD}/${DVISION_ROOT}:$PYTHONPATH

WATERZ_ROOT=waterz_root
WATERZ_REPOSITORY=https://github.com/funkey/waterz
WATERZ_REVISION=d2bede846391c56a54365c13d5b2f2f4e6db4ecd
mkdir ${WATERZ_ROOT}
git clone ${WATERZ_REPOSITORY} ${WATERZ_ROOT}
cd ${WATERZ_ROOT}
git checkout ${WATERZ_REVISION}
mkdir -p .cython/inline
cd ..
export PYTHONPATH=${PWD}/${WATERZ_ROOT}:$PYTHONPATH
# install gunpowder

# assumes that gunpowder package directory and requirements.txt are in build
# context (the complementary Makefile ensures that)
cd gunpowder
cp ../requirements.txt .
pip install -r requirements.txt
# export PYTHONPATH :$PYTHONPATH

# test the container
echo $PWD
cd ..
export PYTHONPATH=${PWD}:$PYTHONPATH
echo $PYTHONPATH
echo $PWD
# python test_environment.py
