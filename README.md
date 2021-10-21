# SQN_tensorflow

This repo is a TensorFlow implementation of **[Semantic Query Network (SQN)](https://arxiv.org/abs/2104.04891)**. For Pytorch implementation, check our **[SQN_pytorch](https://github.com/PointCloudYC/SQN_pytorch)** repo.

**Caution**: currently, this repo  **does not achieve a satisfactory result as the SQN paper reports**. For performance details, check [performance](#performance-on-s3dis) section. 

The repo is still under development, with the aim of reaching the level of performance reported in the SQN paper.

Please open an issue, if you have any comments and suggestions for improving the model performance.

## TODOs

- implement the training strategy mentioned in the Appendix of the paper.
- ablation study
- benchmark weak supervision

## How to run 

The latest codes are tested on two Ubuntu settings:

- Ubuntu 18.04, Nvidia 1080, CUDA 10.1, TensorFlow 1.13 and Python 3.6
- Ubuntu 18.04, Nvidia 3090, CUDA 11.3, TensorFlow 1.13 and Python 3.6


### Clone the repository

```
git clone https://github.com/PointCloudYC/SQN_tensorflow && cd SQN_tensorflow
```

### Setup python environment

create a conda environment

```
conda create -n randlanet python=3.5
source activate randlanet
pip install -r helper_requirements.txt
sh compile_op.sh
```

### Download S3DIS and make a symlink

You can download the S3DIS dataset from [here](https://goo.gl/forms/4SoGp4KtH1jfRqEj2") (4.8 GB). You only need to download the file named `Stanford3dDataset_v1.2.zip`, unzip and move (or link) it to a folder. (same as the RandLA-Net repo setting.)

```
# assume S3DIS dataset is downloaded at /media/yinchao/dataset/S3DIS
ln -s /media/yinchao/dataset/S3DIS ./data/S3DIS/Stanford3dDataset_v1.2_Aligned_Version   
```

### Preprocess the raw dataset

You can use the `s3dis-prepare-sqn.sh` script to prepare the S3DIS dataset with weak labels.

```
# prepare the dataset, each room(Note: each area is preprocessed in the CLoserLook3D code) will result in four files(1 file in the original_ply folder for raw_pc.ply, and 3 files in the input_0.040 for sub_pc.py, sub_pc.kdtree, and project_indices file for each raw point), check data_prepare_s3dis.py for details.

python utils/data_prepare_s3dis.py

# check #rooms in npy format, should be 272 rooms
find *.npy | wc -l
```

The data file structure should look like:

```
<root>
├── ...
├── data
│   └── S3DIS
│       └── Stanford3dDataset_v1.2_Aligned_Version
│           ├── Area_1
│           ├── Area_2
│           ├── Area_3
│           ├── Area_4
│           ├── Area_5
│           └── Area_6
│       └── input_0.040
│       └── original_ply
│       └── weak_label_0.01
└── ...
```

### Compile custom CUDA tf_ops

Only `tf_ops/3d_interpolation` CUDA ops need to be compiled, which will used for three trilinear interpolation.

check the `tf_interpolate_compile.sh`; You may need to tailor the `CUDA_ROOT` and `TF_ROOT` path according to your own system.

```
#/bin/bash
CUDA_ROOT="/usr/local/cuda-10.1"
TF_ROOT="/home/yinchao/miniconda3/envs/DL/lib/python3.6/site-packages/tensorflow"

# TF1.4 (Note: -L ${TF_ROOT} should have a space in between)
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I ${TF_ROOT}/include -I ${CUDA_ROOT}/include -I ${TF_ROOT}/include/external/nsync/public -lcudart -L ${CUDA_ROOT}/lib64/ -L ${TF_ROOT} -ltensorflow_framework -O2 # -D_GLIBCXX_USE_CXX11_ABI=0
```

For more details, check [Charles' PointNet2](https://github.com/charlesq34/pointnet2)

### Train on S3DIS

You can use `run-s3dis-Sqn.sh` script to training multiple settings. The core part is as follows:

```
python -B main_S3DIS_sqn.py --gpu 0 --mode train --test_area 5
python -B main_S3DIS_sqn.py --gpu 0 --mode test --test_area 5

# for cross validation, use the script
# sh jobs_6_fold_cv_s3dis.sh

# evaluate
# TODO
```

For more details to set up the development environment, check [the official RandLA-Net repo](https://github.com/QingyongHu/RandLA-Net).

## Performance on S3DIS

The experiments are still in progress due to my slow GPU.

| Model                           | Weak ratio | Performance (mIoU, %) | Description                                                                                                                                                                                                            |
|---------------------------------|------------|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Official RandLA-Net| 100%| 63.0| Fully supervised method trained with full labels.                                                                                                                                                                      |   |
| Official SQN| 1/1000| 61.4| This official SQN uses additional techniques to improve the performance, our replicaed SQN currently does not investigate this yet. Official SQN does not provide results of S3DIS under the weak ratio of 1/10 and 1/100 |
| Our replicated SQN| 1/10| 52.94| Use RandLA-Net as the encoder and active learning is currently not used.                                                                                                                                                |
| Our replicated SQN| 1/100| 32.96| Use RandLA-Net as the encoder and active learning is currently not used.                                                                                                                                                |
| Our replicated SQN| 1/1000| --| Use RandLA-Net as the encoder and active learning is currently not used.                                                                                                                                                |


## Acknowledgements

Our pytorch codes borrowed a lot from [RandLA-Net](https://github.com/QingyongHu/RandLA-Net) and the custom trilinear interoplation CUDA ops are modified from [Charles Qi's Pointnet2](https://github.com/charlesq34/pointnet2).

## Citation

If you find our work useful in your research, please consider citing:

```
@article{pytorchpointnet++,
    Author = {YIN, Chao},
    Title = {SQN TensorFlow implementation},
    Journal = {https://github.com/PointCloudYC/SQN_tensorflow},
    Year = {2021}
   }

@article{hu2021sqn,
    title={SQN: Weakly-Supervised Semantic Segmentation of Large-Scale 3D Point Clouds with 1000x Fewer Labels},
    author={Hu, Qingyong and Yang, Bo and Fang, Guangchi and Guo, Yulan and Leonardis, Ales and Trigoni, Niki and Markham, Andrew},
    journal={arXiv preprint arXiv:2104.04891},
    year={2021}
  }
```