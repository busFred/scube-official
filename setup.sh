set -e
conda install -c conda-forge -c nvidia/label/cuda-12.1.0 -c pytorch -c pytorch3d "numpy<2.0.0" pytorch=2.2.0 pytorch-cuda=12.1 torchvision=0.17.0 pytorch3d git gitpython ca-certificates certifi openssl gcc_linux-64=11 gxx_linux-64=11 cmake make ninja ffmpeg sparsehash cuda-nvcc cuda-toolkit=12.1 -y
export CUDA_HOME=$CONDA_PREFIX
export CUDA_ROOT=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/include:$CPATH
export MAX_JOBS=$(nproc)  # Use all available CPU cores
export MAKEFLAGS="-j$(nproc)"  # For make-based builds
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)  # For CMake builds
pip install torch-geometric
FORCE_CUDA=1 pip install torch-scatter torch-sparse torch-cluster torch-spline-conv --no-build-isolation --use-pep517
# Install remaining packages
pip install -r requirements.txt
mim install "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"