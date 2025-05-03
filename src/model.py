import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.cuda.amp import autocast

# 3D Inception-Residual Block
class InceptionResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv3d(in_channels, out_channels // 4, kernel_size=1, padding=0)
        self.conv3x3 = nn.Conv3d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv3d(in_channels, out_channels // 4, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv3d(in_channels, out_channels // 4, kernel_size=7, padding=3)
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn = nn.InstanceNorm3d(out_channels)

    def forward(self, x):
        out = torch.cat([self.conv1x1(x), self.conv3x3(x), self.conv5x5(x), self.conv7x7(x)], dim=1)
        out = self.bn(out + self.residual(x))
        return F.relu(out)
# 3D Vision Transformer Encoder Block
class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

# Multi-Scale Fusion Module
class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return F.relu(self.conv(x))

# The Hybrid ViT-CNN-PCR Model
class ViT_CNN_Segmentation(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.enc1 = InceptionResBlock(4, 32)
        self.enc2 = InceptionResBlock(32, 64)
        self.enc3 = InceptionResBlock(64, 128)
        self.enc4 = InceptionResBlock(128, 128)  # Adjusted to 128 channels
        
        # Transformer Bottleneck
        self.flatten = Rearrange('b c d h w -> b (d h w) c')
        self.vit = ViTBlock(128, num_heads=8)
        self.unflatten = Rearrange('b (d h w) c -> b c d h w', d=16, h=16, w=16)
        
        # Decoder
        self.dec3 = MultiScaleFusion(128, 128, 128)
        self.dec2 = MultiScaleFusion(128, 64, 64)
        self.dec1 = MultiScaleFusion(64, 32, 32)
        
        # Output layer
        self.out_conv = nn.Conv3d(32, 3, kernel_size=1)
    
    def forward(self, x):
        # Encoding
        x1 = self.enc1(x)
        x2 = self.enc2(F.max_pool3d(x1, 2))
        x3 = self.enc3(F.max_pool3d(x2, 2))
        x4 = self.enc4(F.max_pool3d(x3, 2))
        
        # Transformer
        vit_features = self.vit(self.flatten(x4))
        vit_features = self.unflatten(vit_features)
        
        # Decoding
        x = self.dec3(vit_features, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)
        
        return self.out_conv(x)

