# MCD-Net: A Lightweight Deep Learning Baseline for Optical-Only Moraine Segmentation

Official implementation of 
**MCD-Net**
, a lightweight deep learning framework for optical-only moraine segmentation, as presented in our IEEE journal paper.

## Overview

MCD-Net is a lightweight deep learning baseline that integrates MobileNetV2, Convolutional Block Attention Module (CBAM), and DeepLabV3+ decoder for moraine segmentation from optical imagery. This work establishes the first reproducible benchmark for optical-only moraine segmentation with a novel dataset of 3,340 annotated high-resolution images.

## Dataset

The MCD Dataset contains 3,340 high-resolution image-mask pairs from Sichuan and Yunnan, China:

- **Images**: 1024×1024 pixels, 0.5-2.0m resolution
- **Classes**: Binary segmentation (background vs. moraine body)
- **Split**: 2,630 training + 293 test images

**Download the dataset from:**
 https://doi.org/10.5281/zenodo.17435106

## Training Steps
1. Place the dataset downloaded from Zenodo into the `dataset` folder.
2. Before training, place the label files in `dataset/Morainse_dataset/SegmentationClass` and the image files in `dataset/Morainse_dataset/JPEGImages`.
3. Run `dataset_annotation.py` to generate the corresponding dataset split text files before training.
4. In the `train.py` file, select the pre-trained weights you want to use (default parameters are already set).
5. Run `train.py` to start training.

## Prediction Steps
This repository provides a trained pth file (`MCDNet_mobilenetv2_best.pth`). Set the relevant paths in `mcdnet_predictor.py`, then select the prediction mode in `predict.py` and run it.
If you want to use your own trained model, please modify the relevant paths accordingly.

## Reference
https://github.com/ggyyzm/pytorch_segmentation

https://github.com/bubbliiiing/deeplabv3-plus-pytorch

https://github.com/bonlime/keras-deeplab-v3-plus


