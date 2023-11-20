# VRSD: Volumetric Silhouette Rendering for Monocular 3D Object Detection without 3D Supervision

![python](https://img.shields.io/badge/Python-3.9-3670A0?style=flat&logo=Python&logoColor=ffdd54)
![pytorch](https://img.shields.io/badge/PyTorch-1.13-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=%23EE4C2C)

## Environment

```bash
conda env create -f environment.yml
```

## Installation

```python
pip install -e .
```

## Data Preparation

### [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/)

#### Transform 3D bounding boxes from the world coordinate system to the camera coordinate system

```bash
python tools/datasets/kitti_360/make_annotations.py
```

#### Check whether the 3D bounding boxes are correctly transformed (optional)

```bash
python tools/datasets/kitti_360/visualize_annotations.py
```

### [KITTI-Raw](https://www.cvlibs.net/datasets/kitti/raw_data.php)

#### Calcurate the extrinsic parameters from OxTS measurements

```bash
python tools/datasets/kitti_360/calculate_camera_params.py
```

#### Make pseudo 2D bounding boxes with InternImage

```bash
cd tools/models/intern_image/detection/docker
bash docker_run.sh
python scripts/inference.py
```

#### Make pseudo segmentation masks with Segment Anything Model

```bash
cd tools/models/segment_anything
python scripts/inference.py
```

## Distributed Training

Please use Slurm, which is a job scheduling system for clusters, on Tokyo Data Center.

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

## Authors

- Hiroki Sakuma (sakuma@sensetime.jp)
