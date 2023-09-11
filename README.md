# Active Learning with Core-set Sampling and Scale-sensitive Loss for 3D Object Detection

# Requirements

## Installation

* Linux
* Python 3.6+
* PyTorch 1.1 or higher 
* CUDA 9.0 or higher

Please refer to [INSTALL.md](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md) for the installation.

## Config

The dataset configs are located within [tools/cfgs/dataset_configs](../tools/cfgs/dataset_configs), and the model configs are located within [tools/cfgs](../tools/cfgs) for different datasets.

# Getting Started

* Download the offical KITTI 3D Object Detection Dataset and organize as follows:
  
  ```
  AL4OD
  ├── data
  │   ├── kitti
  │   │   │── ImageSets
  │   │   │── training
  │   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
  │   │   │── testing
  │   │   │   ├──calib & velodyne & image_2
  ├── pcdet
  ├── tools
  ```
* Generate Info as follows:
  
  ```shell
  python -m pcdet.datasets.kitti.kitti_dataset_AL create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset_AL.yaml
  ```

# Training

* Run following command to start training.
  
  ```shell
  python train_AL.py --cfg_file ${CONFIG_FILE}
  ```

# Acknowledgement

Our  implementation is mainly based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/tree/master), thanks for their wonderful work.
