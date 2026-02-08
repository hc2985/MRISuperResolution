from extract_slices import load_nifti, slice_to_base64, base64_to_slice
from metric import compute_psnr, compute_ssim, score

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

input_dim = (112, 138)
output_dim = (179, 221)
input_layer = 40
output_layer = 200

latent_dim = 100
channels = 1 # suggested default : 1, number of image channels (gray scale)
img_shape = (channels, 112, 138) # (Channels, Image Size(H), Image Size(W))

class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv3d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):    
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate 
        self.dense_layers = nn.Sequential(*modules)    
        self.conv_1x1 = nn.Conv3d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)    
        out = out + x
        return out
    
class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RRDB, self).__init__()
        self.RDB1 = RDB(nChannels, nDenselayer, growthRate)
        self.RDB2 = RDB(nChannels, nDenselayer, growthRate)
        self.RDB3 = RDB(nChannels, nDenselayer, growthRate)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

## CBAM ##

class ChannelAttention3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention3D, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )

    def forward(self, x):
        b, c, d, h, w = x.shape
        avg = F.adaptive_avg_pool3d(x, 1).view(b, c)
        mx = F.adaptive_max_pool3d(x, 1).view(b, c)
        att = torch.sigmoid(self.fc(avg) + self.fc(mx))
        return x * att.view(b, c, 1, 1, 1)

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention3D(channels, reduction)
        self.spatial_att = SpatialAttention3D(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Input: (batch, 1, 40, 112, 138)
        # Output: (batch, 1, 200, 179, 221)
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 1, kernel_size=3, padding=1)
        self.RRDB1 = RRDB(64, 3, 16)
        self.cbam = CBAM(64, reduction=8, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (batch, 1, 40, 112, 138)
        x = self.relu(self.conv1(x))  # (batch, 64, 40, 112, 138)
        x = self.RRDB1(x)  # (batch, 64, 40, 112, 138)
        # Upsample depth: 40 → 200
        x = F.interpolate(x, size=(200, 112, 138), mode='trilinear', align_corners=False)
        # Upsample spatial with bicubic: 112×138 → 179×221
        b, c, d, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        x = F.interpolate(x, size=(179, 221), mode='bicubic', align_corners=False)
        x = x.reshape(b, d, c, 179, 221).permute(0, 2, 1, 3, 4)
        x = self.cbam(x)
        x = torch.tanh(self.conv2(x))  # (batch, 1, 200, 179, 221)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 2D U-Net discriminator operating per-slice
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True))
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True))
        # Decoder with skip connections
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True))
        self.dec1 = nn.Sequential(
            nn.Conv2d(32 + 16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True))
        self.out = nn.Sequential(
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid())

    def forward(self, x):
        # x: (batch, 1, D, H, W) → per-slice 2D
        b, c, d, h, w = x.shape
        x = x.squeeze(1).reshape(b * d, 1, h, w)
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        # Decoder
        d2 = F.interpolate(e3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        out = self.out(d1)
        # Reshape back to 3D
        out = out.reshape(b, d, 1, out.shape[2], out.shape[3]).permute(0, 2, 1, 3, 4)
        return out


# =====================
# Objective Functions
# =====================

def ssim_loss(pred, target):
    """Per-slice SSIM with independent normalization to match evaluation metric."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # pred/target: (batch, 1, D, H, W) → flatten slices to (B*D, H*W)
    b, c, d, h, w = pred.shape
    p = pred.squeeze(1).reshape(b * d, h * w)
    t = target.squeeze(1).reshape(b * d, h * w)

    # Normalize each slice independently to [0, 1]
    p_min = p.min(dim=1, keepdim=True)[0]
    p_max = p.max(dim=1, keepdim=True)[0]
    t_min = t.min(dim=1, keepdim=True)[0]
    t_max = t.max(dim=1, keepdim=True)[0]
    p = (p - p_min) / (p_max - p_min + 1e-8)
    t = (t - t_min) / (t_max - t_min + 1e-8)

    # Per-slice SSIM
    mu1 = p.mean(dim=1)
    mu2 = t.mean(dim=1)
    sigma1_sq = ((p - mu1.unsqueeze(1)) ** 2).mean(dim=1)
    sigma2_sq = ((t - mu2.unsqueeze(1)) ** 2).mean(dim=1)
    sigma12 = ((p - mu1.unsqueeze(1)) * (t - mu2.unsqueeze(1))).mean(dim=1)

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim.mean()


def psnr_loss(pred, target):
    """Per-slice PSNR with independent normalization to match evaluation metric."""
    b, c, d, h, w = pred.shape
    p = pred.squeeze(1).reshape(b * d, h * w)
    t = target.squeeze(1).reshape(b * d, h * w)

    # Normalize each slice independently to [0, 1]
    p_min = p.min(dim=1, keepdim=True)[0]
    p_max = p.max(dim=1, keepdim=True)[0]
    t_min = t.min(dim=1, keepdim=True)[0]
    t_max = t.max(dim=1, keepdim=True)[0]
    p = (p - p_min) / (p_max - p_min + 1e-8)
    t = (t - t_min) / (t_max - t_min + 1e-8)

    mse = ((p - t) ** 2).mean(dim=1)
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    psnr = torch.clamp(psnr, 0, 50)
    return psnr.mean()


# Loss functions
adversarial_loss = nn.BCELoss()


def generator_loss(discriminator, fake, real):
    # Adversarial: fool discriminator into thinking fake is real
    d_fake = discriminator(fake)
    real_label = torch.ones_like(d_fake)
    g_adv = adversarial_loss(d_fake, real_label)

    # Competition metric: 0.5 * SSIM + 0.5 * (PSNR / 50)
    ssim = ssim_loss(fake, real)
    psnr = psnr_loss(fake, real)
    competition_score = 0.5 * ssim + 0.5 * (psnr / 50)

    # Maximize competition_score = minimize (1 - competition_score)
    return g_adv + 40 * (1 - competition_score) 


def discriminator_loss(discriminator, fake, real):
    # Real images → label 1
    d_real = discriminator(real)
    real_label = torch.ones_like(d_real)
    loss_real = adversarial_loss(d_real, real_label)

    # Fake images → label 0 (detach so generator doesn't get gradients)
    d_fake = discriminator(fake.detach())
    fake_label = torch.zeros_like(d_fake)
    loss_fake = adversarial_loss(d_fake, fake_label)

    return (loss_real + loss_fake) / 2

#Epoch [50/50] G_loss: 43.7042 D_loss: 0.5519 to beat
#Epoch [100/100] G_loss: 42.3399 D_loss: 0.5915 to beat
