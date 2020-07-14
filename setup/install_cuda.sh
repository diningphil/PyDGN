#!/bin/bash
conda create -n pydgn python=3.7
conda activate pydgn

echo export OMP_NUM_THREADS=1 >> ~/.bashrc  # to prevent Pytorch from automatically spawning multiple threads

pip install torch==1.4.0
pip install -r setup/requirements.txt

# CUDA 10.1
pip install -r setup/other_requirements_cuda.txt --find-links https://pytorch-geometric.com/whl/torch-1.4.0.html
