from extract_slices import load_nifti, slice_to_base64, base64_to_slice
from metric import compute_ms_ssim, score

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
        self.RRDB_inc = RRDB(base_ch, 2, 16)
        self.down1 = nn.Sequential(
            nn.Conv3d(base_ch, base_ch * 2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.RRDB_down1 = RRDB(base_ch * 2, 2, 16)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bridge = nn.Sequential(
            nn.Conv3d(base_ch * 2, base_ch * 2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.RRDB_bridge = RRDB(base_ch * 2, 2, 16)
        self.upconv1 = nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=2, stride=2, bias=False)
        self.up1 = nn.Sequential(
            nn.Conv3d(base_ch * 3, base_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.RRDB_up = RRDB(base_ch, 2, 16)

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
    def __init__(self):
        super(Generator, self).__init__()
        # Input: (batch, 1, 40, 112, 138)
        # Output: (batch, 1, 200, 179, 221)
        self.unet = Unet3D(in_ch=1, base_ch=48)
        self.cbam = CBAM(48, reduction=8, kernel_size=3)
        self.conv2 = nn.Conv3d(48, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (batch, 1, 40, 112, 138)
        x = self.unet(x)               # (batch, 48, 40, 112, 138)

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

# =====================
# Objective Functions
# =====================

def _gaussian_kernel_2d(size=11, sigma=1.5):
    """Create a 2D Gaussian kernel as a (1,1,size,size) PyTorch tensor."""
    radius = size // 2
    y = torch.arange(-radius, radius + 1, dtype=torch.float32)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)


def ms_ssim_loss(pred, target, weights=None, win_size=11, sigma=1.5):
    """Differentiable MS-SSIM matching the competition metric (per-slice, [0,1]-normalised)."""
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    n_scales = len(weights)

    # (B, 1, D, H, W) → (B*D, 1, H, W)
    b, c, d, h, w = pred.shape
    p = pred.squeeze(1).reshape(b * d, 1, h, w)
    t = target.squeeze(1).reshape(b * d, 1, h, w)

    # Normalize each slice independently to [0, 1]
    p_flat = p.reshape(b * d, -1)
    t_flat = t.reshape(b * d, -1)
    p = (p - p_flat.min(dim=1)[0].view(-1, 1, 1, 1)) / (p_flat.max(dim=1)[0].view(-1, 1, 1, 1) - p_flat.min(dim=1)[0].view(-1, 1, 1, 1) + 1e-8)
    t = (t - t_flat.min(dim=1)[0].view(-1, 1, 1, 1)) / (t_flat.max(dim=1)[0].view(-1, 1, 1, 1) - t_flat.min(dim=1)[0].view(-1, 1, 1, 1) + 1e-8)

    kernel = _gaussian_kernel_2d(win_size, sigma).to(pred.device)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mcs_list = []
    lum_last = None
    for scale in range(n_scales):
        if p.shape[2] < win_size or p.shape[3] < win_size:
            break

        mu1 = F.conv2d(p, kernel)
        mu2 = F.conv2d(t, kernel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

        sigma1_sq = torch.clamp(F.conv2d(p * p, kernel) - mu1_sq, min=0.0)
        sigma2_sq = torch.clamp(F.conv2d(t * t, kernel) - mu2_sq, min=0.0)
        sigma12 = F.conv2d(p * t, kernel) - mu1_mu2

        luminance = (2.0 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
        cs = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

        mcs_list.append(cs.mean())
        if scale == n_scales - 1:
            lum_last = luminance.mean()

        # Downsample 2x (trim to even dims first)
        if scale < n_scales - 1:
            h_cur, w_cur = p.shape[2], p.shape[3]
            p = F.avg_pool2d(p[:, :, :h_cur - h_cur % 2, :w_cur - w_cur % 2], 2)
            t = F.avg_pool2d(t[:, :, :h_cur - h_cur % 2, :w_cur - w_cur % 2], 2)

    n_computed = len(mcs_list)
    if n_computed == 0:
        return torch.tensor(0.0, device=pred.device)

    # Renormalize weights to computed scales
    used_w = weights[:n_computed]
    w_sum = sum(used_w)
    used_w = [ww / w_sum for ww in used_w]

    ms_val = torch.tensor(1.0, device=pred.device)
    for i, cs_val in enumerate(mcs_list):
        cs_c = torch.clamp(cs_val, 0.0, 1.0)
        if i == n_computed - 1 and lum_last is not None:
            lum_c = torch.clamp(lum_last, 0.0, 1.0)
            ms_val = ms_val * (lum_c ** used_w[i]) * (cs_c ** used_w[i])
        else:
            ms_val = ms_val * (cs_c ** used_w[i])

    return ms_val

# Loss functions
def generator_loss(fake, real):
    ms_ssim = ms_ssim_loss(fake, real)
    l1 = F.l1_loss(fake, real)
    return 0.5 * l1 + 0.5 * (1 - ms_ssim)