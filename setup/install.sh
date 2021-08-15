# default pytorch version
PYTORCH_VERSION=1.9.0
PYTORCH_GEOMETRIC_VERSION=1.7.2

# set CUDA variable (defaults to cpu if no argument is provided to the script)
# available options for for pytorch 1.7.2 are cpu, cu102, cu111
CUDA_VERSION=${1:-cpu}

# create virtual environment and activate it
conda create --name pydgn python=3.7 -y
conda activate pydgn

# install pytorch
if [[ "$CUDA_VERSION" == "cpu" ]]; then
  conda install pytorch==${PYTORCH_VERSION} torchvision torchaudio cpuonly -c pytorch -y
elif [[ "$CUDA_VERSION" == 'cu102' ]]; then
  conda install pytorch==${PYTORCH_VERSION} torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
elif [[ "$CUDA_VERSION" == 'cu111' ]]; then
  conda install pytorch==${PYTORCH_VERSION} torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia -y
fi

conda install jupyter -y

# install torch-geometric dependencies
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-geometric==${PYTORCH_GEOMETRIC_VERSION}

echo "Done. Remember to "
echo " 1) append the anaconda/miniconda lib path to the LD_LIBRARY_PATH variable using the export command"
echo "Modify the .bashrc file to make permanent changes."
