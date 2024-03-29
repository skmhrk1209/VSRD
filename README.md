# VSRD: Instance-Aware Volumetric Silhouette Rendering for Weakly Supervised 3D Object Detection

Official Implementation of "VSRD: Instance-Aware Volumetric Silhouette Rendering for Weakly Supervised 3D Object Detection" [CVPR 2024]

![python](https://img.shields.io/badge/Python-3.10-3670A0?style=flat&logo=Python&logoColor=ffdd54)
![pytorch](https://img.shields.io/badge/PyTorch-1.13-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=%23EE4C2C)

## Installation

1. Setup the conda environment.

```bash
conda env create -f environment.yml
```

2. Install this repository.

```python
pip install -e .
```

## Data Preparation

1. Download the [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/download.php) dataset.

2. Transform the 3D bounding boxes from the world coordinate system to the camera coordinate system.

```bash
python tools/datasets/kitti_360/make_annotations.py
```

3. For each target view, sample source frames for multi-view 3D auto-labeling.

```bash
python tools/datasets/kitti_360/sample_annotations.py
```

## Multi-View 3D Auto-Labeling

Our multi-view 3D auto labeling optimizes the 3D bounding boxes and residual signed distance fields (RDF) for each scene. Note that the data is split and distributed to each process, but unlike general distributed training, the gradients are not averaged between processes.

- [Slurm Workload Manager](https://ja.wikipedia.org/wiki/Slurm_Workload_Manager)

```bash
python -m vsrd.distributed.slurm.launch \
    --partition PARTITION \
    --num_nodes NUM_NODES \
    --num_gpus NUM_GPUS \
    scripts/main.py \
        --launcher slurm \
        --config CONFIG \
        --train
```

- [Torch Distributed Launcher](https://pytorch.org/docs/stable/elastic/run.html)

```bash
torchrun \
    --rdzv_backend c10d \
    --rdzv_endpoint HOST_NODE_ADDR \
    --nnodes NUM_NODES \
    --nproc_per_node NUM_GPUS \
    scripts/main.py \
        --launcher torch \
        --config CONFIG \
        --train
```
