# default pytorch version
PYTORCH_VERSION=1.10.2
PYTORCH_GEOMETRIC_VERSION=2.0.3

# set CUDA variable (defaults to cpu if no argument is provided to the script)
CUDA_VERSION=${1:-cpu}

# create virtual environment and activate it
conda create --name pydgn python=3.8 -y
conda activate pydgn

# install pytorch
if [[ "$CUDA_VERSION" == "cpu" ]]; then
  conda install pytorch==${PYTORCH_VERSION} torchvision torchaudio cpuonly -c pytorch -y
elif [[ "$CUDA_VERSION" == 'cu102' ]]; then
  conda install pytorch==${PYTORCH_VERSION} torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
elif [[ "$CUDA_VERSION" == 'cu113' ]]; then
  conda install pytorch==${PYTORCH_VERSION} torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia -y
fi

conda install jupyter -y

conda install pyg==${PYTORCH_GEOMETRIC_VERSION} -c pyg -c conda-forge -c rusty1s -c conda-forge

echo "Done. Remember to "
echo " 1) append the anaconda/miniconda lib path to the LD_LIBRARY_PATH variable using the export command"
echo "Modify the .bashrc file to make permanent changes."
