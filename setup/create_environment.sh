# default pytorch version
PYTHON=python3.8
PYTORCH_VERSION=1.13.0
PYTORCH_GEOMETRIC_VERSION=2.1.0.post1

# set CUDA variable (defaults to cpu if no argument is provided to the script)
CUDA_VERSION=${1:-cpu}

# create virtual environment and activate it
$PYTHON -m pip install --user virtualenv
$PYTHON -m venv ~/.venv/pydgn
source ~/.venv/pydgn/bin/activate

pip install build wheel pytest black

# install pytorch
if [[ "$CUDA_VERSION" == "cpu" ]]; then
  pip3 install torch==${PYTORCH_VERSION} torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
elif [[ "$CUDA_VERSION" == 'cu112' ]]; then
  pip3 install torch==${PYTORCH_VERSION} torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu112
elif [[ "$CUDA_VERSION" == 'cu113' ]]; then
  pip3 install torch==${PYTORCH_VERSION} torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
elif [[ "$CUDA_VERSION" == 'cu116' ]]; then
  pip3 install torch==${PYTORCH_VERSION} torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
elif [[ "$CUDA_VERSION" == 'cu117' ]]; then
  pip3 install torch==${PYTORCH_VERSION} torchvision torchaudio
fi

pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric==${PYTORCH_GEOMETRIC_VERSION} -f https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+${CUDA_VERSION}.html

pip install jupyter
#pip install pydgn
#python -m build && pip uninstall pydgn && pip install dist/pydgn-1.3.0-py3-none-any.whl

echo "Done. Remember to append the CUDA lib path to the LD_LIBRARY_PATH variable using the export command in the .bashrc file"
