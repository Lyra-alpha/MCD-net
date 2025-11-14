# MCD-Net: A Lightweight Deep Learning Baseline for Optical-Only Moraine Segmentation

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)
](https://creativecommons.org/licenses/by/4.0/)
[![Paper](https://img.shields.io/badge/Paper-IEEE-blue.svg)
]()
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-green.svg)
](https://doi.org/10.5281/zenodo.17435106)

Official implementation of 
**MCD-Net**
, a lightweight deep learning framework for optical-only moraine segmentation, as presented in our IEEE journal paper.

## Overview

MCD-Net is a lightweight deep learning baseline that integrates MobileNetV2, Convolutional Block Attention Module (CBAM), and DeepLabV3+ decoder for moraine segmentation from optical imagery. This work establishes the first reproducible benchmark for optical-only moraine segmentation with a novel dataset of 3,340 annotated high-resolution images.

## Dataset

The MCD Dataset contains 3,340 high-resolution image-mask pairs from Sichuan and Yunnan, China:

- **Images**
: 1024×1024 pixels, 0.5-2.0m resolution
- **Classes**
: Binary segmentation (background vs. moraine body)
- **Split**
: 2,630 training + 293 test images
- **Availability**: [Zenodo Dataset](https://doi.org/10.5281/zenodo.17435106)

**Download the dataset from:**
 https://doi.org/10.5281/zenodo.17435106

## Reference
[https://github.com/ggyyzm/pytorch_segmentation  ](https://github.com/bubbliiiing/deeplabv3-plus-pytorch)
