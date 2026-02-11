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
    

class Unet3D(nn.Module):
    """Lightweight 3D U-Net that returns `base_ch` feature channels.
    It's intentionally small: one down/up step to keep spatial/depth sizes
    compatible with the rest of the generator pipeline.
    """
    def __init__(self, in_ch=1, base_ch=64):
        super(Unet3D, self).__init__()
        self.inc = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.RRDB_inc = RRDB(base_ch, 3, 16)
        self.down1 = nn.Sequential(
            nn.Conv3d(base_ch, base_ch * 2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.RRDB_down1 = RRDB(base_ch * 2, 3, 16)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bridge = nn.Sequential(
            nn.Conv3d(base_ch * 2, base_ch * 2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.RRDB_bridge = RRDB(base_ch * 2, 3, 16)
        self.upconv1 = nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=2, stride=2, bias=False)
        self.up1 = nn.Sequential(
            nn.Conv3d(base_ch * 2, base_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.RRDB_up = RRDB(base_ch, 3, 16)
        self.RRDB1 = RRDB(base_ch, 3, 16)

    def forward(self, x):
        # x: (B, 1, D, H, W)
        x1 = self.inc(x)            # (B, base_ch, D, H, W)
        x1 = self.RRDB_inc(x1)
        x2 = self.down1(x1)         # (B, base_ch*2, D, H, W)
        x2 = self.RRDB_down1(x2)
        x2p = self.pool(x2)         # (B, base_ch*2, D/2, H/2, W/2)
        xb = self.bridge(x2p)       # (B, base_ch*2, ...)
        xb = self.RRDB_bridge(xb)
        xu = self.upconv1(xb)       # (B, base_ch, D, H, W) -- ideally matches x2 spatially

        # Align skip feature `x2` to `xu` if necessary (handles odd dims)
        if xu.shape[2:] != x2.shape[2:]:
            dz = (x2.shape[2] - xu.shape[2]) // 2
            dy = (x2.shape[3] - xu.shape[3]) // 2
            dx = (x2.shape[4] - xu.shape[4]) // 2
            x2_c = x2[:, :, dz:dz + xu.shape[2], dy:dy + xu.shape[3], dx:dx + xu.shape[4]]
        else:
            x2_c = x2

        xcat = torch.cat([xu, x2_c], dim=1)
        xout = self.up1(xcat)       # (B, base_ch, D, H, W)
        xout = self.RRDB_up(xout)
        return xout
    
class Generator(nn.Module):
    def __init__(self, mode='rrdb'):
        super(Generator, self).__init__()
        # Input: (batch, 1, 40, 112, 138)
        # Output: (batch, 1, 200, 179, 221)
        # mode: 'rrdb' (default) uses the existing RRDB pipeline
        #       'unet'         uses a lightweight 3D U-Net backbone
        self.mode = mode
        # keep conv1 for RRDB path compatibility
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.RRDB1 = RRDB(64, 3, 16)
        # U-Net path
        if self.mode == 'unet':
            self.unet = Unet3D(in_ch=1, base_ch=64)

        self.cbam = CBAM(64, reduction=8, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (batch, 1, 40, 112, 138)
        if self.mode == 'unet':
            x = self.unet(x)               # (batch, 64, 40, 112, 138)
        else:
            x = self.relu(self.conv1(x))  # (batch, 64, 40, 112, 138)
            x = self.RRDB1(x)             # (batch, 64, 40, 112, 138)

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
    """PatchGAN-style discriminator applied per-slice (2D).
    Takes input (B,1,D,H,W) and returns per-slice patch predictions
    shaped (B,1,D,H_out,W_out).
    """
    def __init__(self, in_ch=1, ndf=64):
        super(Discriminator, self).__init__()
        # PatchGAN 4-layer conv net (from Pix2Pix-style)
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, 1, D, H, W) → per-slice 2D
        b, c, d, h, w = x.shape
        x = x.squeeze(1).reshape(b * d, 1, h, w)
        out = self.model(x)  # shape (b*d, 1, H_out, W_out)
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
    return g_adv + 25 * (1 - competition_score) 


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