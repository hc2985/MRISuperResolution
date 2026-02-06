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

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Input: (batch, 1, 40, 112, 138)
        # Output: (batch, 1, 200, 179, 221)
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 1, kernel_size=3, padding=1)
        self.RDB1 = RDB(64, 3, 16)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (batch, 1, 40, 112, 138)
        x = self.relu(self.conv1(x))  # (batch, 64, 40, 112, 138)
        x = self.RDB1(x)  # (batch, 64, 40, 112, 138)

        # Upsample depth: 40 → 200
        x = F.interpolate(x, size=(200, 112, 138), mode='trilinear', align_corners=False)

        # Upsample spatial with bicubic: 112×138 → 179×221
        b, c, d, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        x = F.interpolate(x, size=(179, 221), mode='bicubic', align_corners=False)
        x = x.reshape(b, d, c, 179, 221).permute(0, 2, 1, 3, 4)

        x = torch.tanh(self.conv2(x))  # (batch, 1, 200, 179, 221)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Input: (batch, 1, 200, 179, 221)
        # Output: (batch, 1, 25, 23, 28) — per-patch real/fake

        self.model = nn.Sequential(
            # (1, 200, 179, 221) → (64, 100, 90, 111)
            nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # (64, 100, 90, 111) → (1, 50, 45, 56)
            nn.Conv3d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)  # (batch, 1, 50, 45, 56)


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
    return g_adv + 100 * (1 - competition_score)


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
