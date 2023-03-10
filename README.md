# The Auto-Bio Challenge: the final project of PKU Computer Vision course
## Firstly ...
This repository is forked and modified from the Trans2Seg official repository, the environment, installation and using method are almost the same. However there are several different points:
* The dataset is our own AUtoBio dataset, you should place the `data` folder in the correct position in the `data` in the root path. 
* When training and testing the model, you should use the `train_autobio.py` in the `tools` folder, and other parameters are the same to use. (I mean the command in the terminal)
* In the original repository, the `demo.py` cannot be used but here you can use the `demo.py` to conduct inference, the using method is the same as training. 
****
## Introduction
This repository contains the data and code for IJCAI 2021 paper [Segmenting transparent object in the wild with transformer](https://arxiv.org/abs/2101.08461).


## Environments

- python 3
- torch = 1.4.0
- torchvision
- pyyaml
- Pillow
- numpy

## INSTALL

```
python setup.py develop --user
```

<!-- ## Data Preparation
1. create dirs './datasets/transparent/Trans10K_v2' 
2. put the train/validation/test data under './datasets/transparent/Trans10K_v2'. 
Data Structure is shown below.
```
Trans10K_v2
├── test
│   ├── images
│   └── masks_12
├── train
│   ├── images
│   └── masks_12
└── validation
    ├── images
    └── masks_12
```
Download Dataset: [Google Drive](https://drive.google.com/file/d/1YzAAMY8xfL9BMTIDU-nFC3dcGbSIBPu5/view?usp=sharing).
[Baidu Drive](https://pan.baidu.com/s/1P-2l-Q2brbnwRd2kXi--Dg). code: oqms -->

## Network Define
The code of Network pipeline is in `segmentron/models/trans2seg.py`.

The code of Transformer Encoder-Decoder is in `segmentron/modules/transformer.py`.

## Train
Our experiments are based on one machine with 8 V100 GPUs with 32g memory, about 1 hour training time.

```
bash tools/dist_train.sh $CONFIG-FILE $GPUS
```

For example:
```
bash tools/dist_train.sh configs/trans10kv2/trans2seg/trans2seg_medium.yaml 8
```

## Test
```
bash tools/dist_train.sh $CONFIG-FILE $GPUS --test TEST.TEST_MODEL_PATH $MODEL_PATH
```


## Citations
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.

```
@article{xie2021segmenting,
  title={Segmenting transparent object in the wild with transformer},
  author={Xie, Enze and Wang, Wenjia and Wang, Wenhai and Sun, Peize and Xu, Hang and Liang, Ding and Luo, Ping},
  journal={arXiv preprint arXiv:2101.08461},
  year={2021}
}
```
