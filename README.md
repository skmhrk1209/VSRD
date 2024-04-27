# VSRD: Instance-Aware Volumetric Silhouette Rendering for Weakly Supervised 3D Object Detection

![python](https://img.shields.io/badge/Python-3.10-3670A0?style=flat&logo=Python&logoColor=ffdd54)
![pytorch](https://img.shields.io/badge/PyTorch-1.13-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=%23EE4C2C)

The official Implementation of ["VSRD: Instance-Aware Volumetric Silhouette Rendering for Weakly Supervised 3D Object Detection" [CVPR 2024]](https://arxiv.org/abs/2404.00149)

https://github.com/skmhrk1209/VSRD/assets/29158616/fc64e7dd-2bb2-4719-b662-cb1e16ce7644

## Installation

1. Setup the conda environment.

```bash
conda env create -f environment.yaml
```

2. Install this repository.

```bash
pip install -e .
```

## Data Preparation

1. Download the [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/download.php) dataset.

    - Perspective Images [[download]](https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/a1d81d9f7fc7195c937f9ad12e2a2c66441ecb4e/download_2d_perspective.zip)
    - Instance Masks [[download](https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/ed180d24c0a144f2f1ac71c2c655a3e986517ed8/data_2d_semantics.zip)]
    - 3D Bounding Boxes [[download]](https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/ffa164387078f48a20f0188aa31b0384bb19ce60/data_3d_bboxes.zip)
    - Camera Parameters [[download]](https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/384509ed5413ccc81328cf8c55cc6af078b8c444/calibration.zip)
    - Camera Poses [[download]](https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/89a6bae3c8a6f789e12de4807fc1e8fdcf182cf4/data_poses.zip)

2. Make a JSON annotation file for each frame.

    - Frames without camera poses are excluded.
    - Frames without instance masks are exluded.
    - 3D bounding boxes are transformed from the world coordinate system to each camera coordinate system.

```bash
python tools/datasets/kitti_360/make_annotations.py
```

3. (Optional) Visualize the annotations to check whether the 3D bounding boxes were successfully transformed.

```bash
python tools/datasets/kitti_360/visualize_annotations.py
```

4. Sample source frames for each target frame.

    - Please refer to the supplementary material for how to sample source frames.
    - Target frames without at least 16 source frames are excluded.
    - Target frames with the same set of instance IDs are grouped together.
    - Only one target frame for each instance group is labeled by VSRD.
    - The pseudo labels for each target frame are shared with all the frames in the same instance group.

```bash
python tools/datasets/kitti_360/sample_annotations.py
```

## Multi-View 3D Auto-Labeling

VSRD optimizes the 3D bounding boxes and residual signed distance fields (RDF) for each target frame. The optimized 3D bounding boxes can be used as pseudo labels for training of any 3D object detectors.

### Distributed Training

Sampled target frames in each sequence are split and distributed across multiple processes, each of which processes the chunk independently. Note that gradients are not averaged between processes unlike general distributed training. Please run [main.py](scripts/main.py) with the corresponding configuration file for each sequence as follows:

- [Slurm](https://slurm.schedmd.com/documentation.html)

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

- [Torchrun](https://pytorch.org/docs/stable/elastic/run.html)

```bash
torchrun \
    --rdzv_backend c10d \
    --rdzv_endpoint HOST_NODE_ADDR \
    --nnodes NUM_NODES \
    --nproc_per_node NUM_GPUS \
    scripts/main.py \
        --launcher torchrun \
        --config CONFIG \
        --train
```

## Pseudo Label Preparation

1. Make a JSON pseudo label file for each target frame from the checkpoint.

    - The pseudo labels for each target frame are shared with all the frames in the same instance group.
    - The pseudo labels for each target frame are transformed from the target camera coordinate system to the other camera coordinate systems.

```bash
python tools/datasets/kitti_360/make_predictions.py
```

2. (Optional) Visualize the pseudo labels to check whether the 3D bounding boxes were successfully transformed.

```bash
python tools/datasets/kitti_360/visualize_predictions.py
```

3. Convert the pseudo labels from our own JSON format to the KITTI format to make use of existing training frameworks like [MMDetection3D](https://github.com/open-mmlab/mmdetection3d).

```bash
python tools/datasets/kitti_360/convert_predictions.py
```

## License

VSRD is released under the MIT license.

## Citation

```bibtex
@article{liu2024vsrd,
title={VSRD: Instance-Aware Volumetric Silhouette Rendering for Weakly Supervised 3D Object Detection},
author={Liu, Zihua and Sakuma, Hiroki and Okutomi, Masatoshi},
journal={arXiv preprint arXiv:2404.00149},
year={2024}
}
```
