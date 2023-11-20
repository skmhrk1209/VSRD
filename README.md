# VSRD: Instance-Aware Volumetric Silhouette Rendering for Weakly Supervised 3D Object Detection

![python](https://img.shields.io/badge/Python-3.9-3670A0?style=flat&logo=Python&logoColor=ffdd54)
![pytorch](https://img.shields.io/badge/PyTorch-1.13-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=%23EE4C2C)

## Installation

### Setup the environment

```bash
conda env create -f environment.yml
```

### Install VSRD

```python
pip install -e .
```

## Data Preparation

### Download [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/)

### Transform 3D bounding boxes from the world coordinate system to the camera coordinate system

```bash
python tools/datasets/kitti_360/make_annotations.py
```

### Check whether the 3D bounding boxes are correctly transformed (optional)

```bash
python tools/datasets/kitti_360/visualize_annotations.py
```

## Distributed Training

### [Slurm Workload Manager](https://ja.wikipedia.org/wiki/Slurm_Workload_Manager)

```bash
python -m vsrd.distributed.slurm.launch \
    --partition PARTITION \
    --num_nodes NUM_NODES \
    --num_gpus NUM_GPUS \
    scripts/main.py \
        --launcher slurm \
        --config CONFIG \
        --train \
        --eval
```

### [Torch Distributed Launcher](https://pytorch.org/docs/stable/elastic/run.html)

```bash
torchrun \
    --rdzv_backend c10d \
    --rdzv_endpoint HOST_NODE_ADDR \
    --nnodes NUM_NODES \
    --nproc_per_node NUM_GPUS \
    scripts/main.py \
        --launcher torch \
        --config CONFIG \
        --train \
        --eval
```
