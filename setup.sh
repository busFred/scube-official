set -e
# Set CUDA environment with correct paths for conda installation
export CUDA_HOME=$CONDA_PREFIX
export CUDA_ROOT=$CONDA_PREFIX
# Set parallel build options
export MAX_JOBS=$(nproc)
export MAKEFLAGS="-j$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
# CRITICAL: Add conda's CUDA header paths to compilation flags
export CPPFLAGS="-I$CONDA_PREFIX/targets/x86_64-linux/include -I$CONDA_PREFIX/include $CPPFLAGS"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include:$CPATH"
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
# Install everything with NumPy constraint from the start
conda install -c conda-forge -c nvidia/label/cuda-12.1.0 -c pytorch -c pytorch3d \
    pytorch=2.2.0 pytorch-cuda=12.1 torchvision=0.17.0 pytorch3d \
    "numpy<2.0.0" \
    git gitpython ca-certificates certifi openssl gcc_linux-64=11 gxx_linux-64=11 \
    cmake make ninja ffmpeg sparsehash cuda-nvcc cuda-toolkit=12.1 -y
# Install PyTorch Geometric with correct CUDA paths
pip install torch-geometric
# Install PyTorch Geometric extensions - should now find CUDA headers
FORCE_CUDA=1 pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
# Install remaining pip packages
pip install -r requirements.txt
echo "Installation complete!"