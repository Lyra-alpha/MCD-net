# MCD-Net: A Lightweight Deep Learning Baseline for Optical-Only Moraine Segmentation

Official implementation of **MCD-Net**, a lightweight deep learning framework that integrates **MobileNetV2**, **CBAM (Convolutional Block Attention Module)**, and **DeepLabV3+** decoder for moraine segmentation from optical imagery.

Pre-trained Model: `model_data/MCDNet_mobilenetv2_best.pth`

---

## Repository Structure

```
MCD-Net/
├── dataset/
│   └── Moraine_dataset/
│       ├── JPEGImages/               # Place your .jpg images here
│       ├── SegmentationClass/        # Place your .png masks here
│       └── ImageSets/Segmentation/   # Auto-generated txt splits
├── nets/                           
│   ├── attention.py                 
│   ├── deeplabv3_plus.py            
│   ├── deeplabv3_training.py       
│   └── mobilenetv2.py              
├── utils/                           
│   ├── callbacks.py                  
│   ├── dataloader.py             
│   ├── utils.py                   
│   ├── utils_fit.py               
│   └── utils_metrics.py          
├── model_data/                  
│   └── MCDNet_mobilenetv2_best.pth                                                                    
├── dataset_annotation.py           
├── train.py                       
├── mcdnet_predictor.py            
├── predict.py                      
├── get_miou.py                     
├── requirements.txt             
└── README.md                     
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Lyra-alpha/MCD-Net.git
cd MCD-Net
```

### 2. Create environment & install dependencies
```bash
conda create -n mcdnet python=3.9 -y
conda activate mcdnet
pip install -r requirements.txt
```

### 3. Verify installation
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

---

## Environment

| Requirement | Version Tested |
|-------------|---------------|
| Python | 3.9 |
| PyTorch | 1.12+ |
| CUDA | 11.6 (GPU recommended) |
| OS | Ubuntu 20.04 / Windows 10/11 |

GPU Memory: >= 6 GB VRAM (batch_size=8)  

---

## Training

### Quick Start
```bash
python train.py
```

### Key Configuration (in `train.py`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_classes` | 2 | Background + Moraine |
| `backbone` | mobilenet | Only MobileNetV2 supported |
| `use_attention` | True | Enable CBAM attention modules |
| `downsample_factor` | 16 | Output stride (8 or 16) |
| `Freeze_Epoch / UnFreeze_Epoch` | 100 / 200 | Two-stage training with SGD |
| `Init_lr` | 7e-3 | Initial learning rate |
| `batch_size` (Freeze) | 8 | Batch size during frozen backbone |
| `batch_size` (Unfreeze) | 4 | Batch size during full training |

Training outputs are saved in `logs/`:
- `best_epoch_weights.pth` -- Best validation loss
- `last_epoch_weights.pth` -- Final epoch
- `loss_*.png`, `epoch_miou.png` -- Loss and mIoU curves

---

## Prediction

### Single Image Prediction
```bash
python predict.py
# Set mode = "predict" in predict.py
# Enter image path when prompted
```

### Batch Folder Prediction (default)
```bash
python predict.py
# Set mode = "dir_predict" in predict.py
# Images from: dataset/Moraine_dataset/JPEGImages/
# Results saved to: img_out/
```

To use your own trained model, update `model_path` in `mcdnet_predictor.py` or pass it as a keyword argument.

---

## Evaluation (mIoU, F1, Precision, Recall)

Evaluate on the validation set:
```bash
python get_miou.py
```

Results saved in `miou_out/`:
- `mIoU.png`, `mPA.png`, `Recall.png`, `Precision.png`, `F1.png`
- `confusion_matrix.csv`
- `PixelAccuracy.png`

---

## Quick Test Example

```bash
# 1. Run data splitting
python dataset_annotation.py

# 2. Start training (2 epochs for sanity check)
#    In train.py, set: UnFreeze_Epoch = 2
python train.py

# 3. Evaluate
python get_miou.py

# 4. Predict on validation images
#    In predict.py, set:
#      mode = "dir_predict"
#      dir_origin_path = "dataset/Moraine_dataset/JPEGImages"
python predict.py
```

---

## Pre-trained Model

| File | Description |
|------|-------------|
| `model_data/MCDNet_mobilenetv2_best.pth` | Best checkpoint (provided) |
| `model_data/mobilenet_v2.pth` | ImageNet-pretrained MobileNetV2 (auto-download) |

To use the pre-trained model for inference:
```python
from mcdnet_predictor import MCDNetPredictor

predictor = MCDNetPredictor(
    model_path='model_data/MCDNet_mobilenetv2_best.pth',
    num_classes=2,
    use_attention=True
)
```

---


## References

- [DeepLabV3+ PyTorch Implementation](https://github.com/bubbliiiing/deeplabv3-plus-pytorch)
- [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

https://github.com/bubbliiiing/deeplabv3-plus-pytorch

https://github.com/bonlime/keras-deeplab-v3-plus


