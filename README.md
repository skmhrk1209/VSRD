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

    Only the following data are required.

    - Left perspective images (124 GB)
    - Left instance masks (2.2 GB)
    - 3D bounding boxes (420 MB)
    - Camera parameters (28 KB)
    - Camera poses (28 MB)

    Make sure the directory structure is the same as below:

    ```bash
    KITTI-360
    ├── calibration         # camera parameters
    ├── data_2d_raw         # perspective images
    ├── data_2d_semantics   # instance masks
    ├── data_3d_bboxes      # 3D bounding boxes
    └── data_poses          # camera poses
    ```

2. Make a JSON annotation file for each frame.

    ```bash
    python tools/kitti_360/make_annotations.py \
        --root_dirname ROOT_DIRNAME \
        --num_workers NUM_WORKERS
    ```

    A directory named `annotations` will be created as follows.

    ```bash
    KITTI-360
    ├── annotations         # per-frame annotations
    ├── calibration         # camera parameters
    ├── data_2d_raw         # perspective images
    ├── data_2d_semantics   # instance masks
    ├── data_3d_bboxes      # 3D bounding boxes
    └── data_poses          # camera poses
    ```

    Note that the following frames are excluded.

    - Frames without camera poses
    - Frames without instance masks

3. (Optional) Visualize the annotations to make sure the previous step has been completed successfully.

    ```bash
    python tools/kitti_360/visualize_annotations.py \
        --root_dirname ROOT_DIRNAME \
        --out_dirname OUT_DIRNAME \
        --num_workers NUM_WORKERS
    ```

4. Sample target frames subject to optimization by VSRD along with the corresponding source frames.

    ```bash
    python tools/kitti_360/sample_annotations.py \
        --root_dirname ROOT_DIRNAME \
        --num_workers NUM_WORKERS
    ```

    A directory named `filenames` will be created as follows.

    ```bash
    KITTI-360
    ├── annotations         # per-frame annotations
    ├── calibration         # camera parameters
    ├── data_2d_raw         # perspective images
    ├── data_2d_semantics   # instance masks
    ├── data_3d_bboxes      # 3D bounding boxes
    ├── data_poses          # camera poses
    └── filenames           # sampled filenames
    ```

    Please refer to the supplementary material for how to sample source frames. For efficiency, we use not all but some frames as target frames for optimization by VSRD as follows:

    1. Frames with the same set of instance IDs are grouped.
    2. Only one frame is sampled as a target frame for each instance group.
    3. Pseudo labels for each target frame are shared with all the frames in the same instance group.

    We split all the sequences into training, validation, and test sets. The number of target frames subject to optimization by VSRD and the number of labeled frames are as follows:

    | Sequence                   | Split      | # Target Frames | # Labeled Frames |
    | :------------------------- | :--------- | --------------: | ---------------: |
    | 2013_05_28_drive_0000_sync | Training   | 2562            | 9666             |
    | 2013_05_28_drive_0002_sync | Training   | 748             | 7569             |
    | 2013_05_28_drive_0003_sync | Validation | 32              | 238              |
    | 2013_05_28_drive_0004_sync | Training   | 658             | 5608             |
    | 2013_05_28_drive_0005_sync | Training   | 408             | 4103             |
    | 2013_05_28_drive_0006_sync | Training   | 745             | 6982             |
    | 2013_05_28_drive_0007_sync | Validation | 64              | 877              |
    | 2013_05_28_drive_0009_sync | Training   | 1780            | 10250            |
    | 2013_05_28_drive_0010_sync | Test       | 908             | 2459             |

## Multi-View 3D Auto-Labeling

VSRD optimizes the 3D bounding boxes and residual signed distance fields (RDF) for each target frame. The optimized 3D bounding boxes can be used as pseudo labels for training of any 3D object detectors.

### Distributed Training

Sampled target frames in each sequence are split and distributed across multiple processes, each of which processes the chunk independently. It takes about 15 minutes on V100 to label each frame. Note that gradients are not averaged between processes unlike general distributed training. Please run [main.py](scripts/main.py) with the corresponding configuration file for each sequence as follows:

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

    ```bash
    python tools/kitti_360/make_predictions.py \
        --root_dirname ROOT_DIRNAME \
        --ckpt_dirname CKPT_DIRNAME \
        --num_workers NUM_WORKERS
    ```

2. (Optional) Visualize the pseudo labels to make sure the previous step has been completed successfully.

    ```bash
    python tools/kitti_360/visualize_predictions.py \
        --root_dirname ROOT_DIRNAME \
        --ckpt_dirname CKPT_DIRNAME \
        --out_dirname OUT_DIRNAME \
        --num_workers NUM_WORKERS
    ```

3. Convert the pseudo labels from our custom JSON format to the KITTI format to utilize existing training frameworks such as [MMDetection3D](https://github.com/open-mmlab/mmdetection3d).

    ```bash
    python tools/kitti_360/convert_predictions.py \
        --root_dirname ROOT_DIRNAME \
        --ckpt_dirname CKPT_DIRNAME \
        --num_workers NUM_WORKERS
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
