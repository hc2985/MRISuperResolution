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

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Input: (batch, 1, 40, 112, 138)
        # Output: (batch, 1, 200, 179, 221)
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (batch, 1, 40, 112, 138)
        x = self.relu(self.conv1(x))  # (batch, 64, 40, 112, 138)

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
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = pred.mean()
    mu2 = target.mean()

    sigma1_sq = ((pred - mu1) ** 2).mean()
    sigma2_sq = ((target - mu2) ** 2).mean()
    sigma12 = ((pred - mu1) * (target - mu2)).mean()

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim


def psnr_loss(pred, target):
    mse = ((pred - target) ** 2).mean()
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    return torch.clamp(psnr, 0, 50)


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
