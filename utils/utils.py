import random
import datetime
import numpy as np
import torch
from PIL import Image
import os
from torch.hub import load_state_dict_from_url

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh
    
#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def preprocess_input(image):
    image /= 255.0
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_mobilenet_weights(model_dir="./model_data"):
    """
    下载MobileNetV2预训练权重
    :param model_dir: 权重保存目录
    :return: 权重文件路径
    """
    # 确保目录存在
    os.makedirs(model_dir, exist_ok=True)
    
    # MobileNetV2预训练权重URL
    mobilenet_urls = [
        'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'  # 官方PyTorch模型
    ]
    
    filename = "mobilenet_v2.pth.tar"
    save_path = os.path.join(model_dir, filename)
    
    # 检查权重文件是否已存在
    if os.path.exists(save_path):
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] MobileNetV2权重已存在: {save_path}")
        return save_path
    
    # 尝试下载权重
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 正在下载MobileNetV2预训练权重...")
    
    for url in mobilenet_urls:
        try:
            print(f"尝试下载: {url}")
            state_dict = load_state_dict_from_url(url, model_dir=model_dir, progress=True)
            
            # 保存权重
            torch.save(state_dict, save_path)
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] MobileNetV2权重下载成功: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"下载失败: {str(e)}")
            continue
    
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 所有下载尝试均失败")
    return None

# 为了保持兼容性，保留原函数名但只支持mobilenet
def download_weights(backbone, model_dir="./model_data"):
    """
    下载预训练权重（仅支持MobileNetV2）
    """
    if backbone != "mobilenet":
        print(f"警告: 只支持'mobilenet'，忽略请求的 '{backbone}'")
        return None
    
    return download_mobilenet_weights(model_dir)
