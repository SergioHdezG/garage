#!/bin/bash

# Create environment
source ~/miniconda3/etc/profile.d/conda.sh
cd ..
conda env create -f environment.yml
conda activate garage
pip install -e '.[all,dev]'

# install gym-miniworld
cd ..
git clone https://github.com/CarlosGual/gym-miniworld.git
cd gym-miniworld
pip install -e .
cd ..
