# nets/cbam.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction_ratio=16, use_max_pool=True):
        super(CBAM, self).__init__()
        self.use_max_pool = use_max_pool
        
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if use_max_pool:
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        )
        
        # 空间注意力
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力
        avg_out = self.mlp(self.avg_pool(x))
        if self.use_max_pool:
            max_out = self.mlp(self.max_pool(x))
            channel_att = self.sigmoid(avg_out + max_out)
        else:
            channel_att = self.sigmoid(avg_out)
            
        x_channel = x * channel_att
        
        # 空间注意力
        avg_out = torch.mean(x_channel, dim=1, keepdim=True)
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_concat = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.spatial_conv(spatial_concat))
        
        x_final = x_channel * spatial_att
        
        return x_final