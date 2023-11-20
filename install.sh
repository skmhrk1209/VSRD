# Download the NVIDIA driver 510.108.03
# Product Type: Data Center / Tesla
# Product Series: V-Series
# Product: Tesla V100
# Operating System: Linux 64-bit
# CUDA Toolkit: 11.6
# https://www.nvidia.com/Download/index.aspx?lang=en-us

conda create -y -n vsrd-pytorch-1.13.0-cuda-11.6 python=3.10

conda activate vsrd-pytorch-1.13.0-cuda-11.6

conda install -y mamba

mamba install -y \
    pytorch==1.13.0 \
    torchvision==0.14.0 \
    pytorch-cuda=11.6 \
    -c pytorch \
    -c nvidia

mamba install -y \
    timm \
    numba \
    boost \
    bokeh \
    mpich \
    mpi4py \
    opencv \
    kornia \
    inflection \
    conda-pack \
    tensorboard \
    pycocotools \
    transformers \
    scikit-image \

mamba install -y \
    ninja \
    gxx==10.4.0 \
    cudatoolkit-dev=11.6 \

pip install nerfacc -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-1.13.0_cu116.html
pip install git+https://github.com/autonomousvision/kitti360Scripts.git

conda-pack
