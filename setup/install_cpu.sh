#!/bin/bash
conda create -n pydgn python=3.7
conda activate pydgn

echo export OMP_NUM_THREADS=1 >> ~/.bashrc  # to prevent Pytorch from automatically spawning multiple threads

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    pip install torch==1.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac OSX
    pip install torch==1.4.0
fi

pip install -r setup/requirements.txt
pip install -r setup/other_requirements_cpu.txt --find-links https://pytorch-geometric.com/whl/torch-1.4.0.html
