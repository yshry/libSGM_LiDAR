# Stereo-LiDAR Fusion by Semi-Global Matching With Discrete Disparity-Matching Cost and Semidensification
This repository contains the code for our IEEE RA-L paper `Stereo-LiDAR Fusion by Semi-Global Matching with Discrete Disparity-Matching Cost and Semidensification` [[IEEE Xplore](https://doi.org/10.1109/LRA.2025.3552236)], [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10930562)], [[arxiv](http://arxiv.org/abs/2504.05148)].

The implementation of our model part is heavily borrowed from [libSGM](https://github.com/fixstars/libSGM). 

## Citing
If you find this code useful, please consider to cite our work.

```
@ARTICLE{10930562,
  author={Yao, Yasuhiro and Ishikawa, Ryoichi and Oishi, Takeshi},
  journal={IEEE Robotics and Automation Letters}, 
  title={Stereo-LiDAR Fusion by Semi-Global Matching With Discrete Disparity-Matching Cost and Semidensification}, 
  year={2025},
  volume={10},
  number={5},
  pages={4548-4555},
  doi={10.1109/LRA.2025.3552236}}
```

## Installation

### Environment

CUDA 11.8

### Build
Build a docker image using Dockerfile in the docker directory.
```shell
cd docker
docker build -t IMAGENAME .
```

Run the docker image with mounting this directory.

In the container, build the source code.
```shell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

### Execution
#### Full pipeline (SDSGM)
You can execute the full pipeline (SDSGM) with sample data upon successful build using the following command.
```shell
./build/sample/spread_sgm_image ./sample_data/left.png ./sample_data/right.png ./sample_data/sparse.png
```
It outputs the following files.
1. test.png: output disparity map as 16-bit monochrome PNG image.
2. test_color.png: output disparity map for visualization.
3. t.txt: text file with the processing time in milliseconds.

#### Only semidensification
You can execute only the semidensification method with sample data by the following command.
```shell
./build/sample/spread_sgm_image ./sample_data/left.png ./sample_data/right.png ./sample_data/sparse.png --sd_only
```

#### SGM with discrete disparity matching cost (DSGM) 
You can execute DSGM without semidensification using the following command.
```shell
./build/sample/spread_sgm_image ./sample_data/left.png ./sample_data/right.png ./sample_data/sparse.png --no_sd
```

### Data
#### Format
- Input images:
    - The left and right images must be color or monochrome images of the same size.
- Input and output disparity maps:
    - 16-bit monochrome PNG file of the exact dimensions as the left and right images.
    - The format is used in the KITTI dataset. Please refer to the dataset and development kit for more details. [KITTI dataset](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

#### Evaluation dataset
KITTI 141 dataset evaluated in our paper is available via [XuelianCheng/LidarStereoNet](https://github.com/XuelianCheng/LidarStereoNet/tree/master?tab=readme-ov-file#validation-dataset).

## Lisence
This software includes the work below that is distributed under the Apache License 2.0.  
> [libSGM](https://github.com/fixstars/libSGM)  
> Copyright (c) Fixstars Corporation  
> Licensed under the Apache License, Version 2.0  
> http://www.apache.org/licenses/LICENSE-2.0  
