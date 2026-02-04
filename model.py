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


    def model():
        pass

    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            
            def block(input_features, output_features, normalize=True):
                layers = [nn.Linear(input_features, output_features)]
                if normalize: # Default
                    layers.append(nn.BatchNorm1d(output_features, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True)) # inplace=True : modify the input directly. It can slightly decrease the memory usage.
                return layers # return list of layers
            
            self.model = nn.Sequential(
                *block(latent_dim, 128, normalize=False), # Asterisk('*') in front of block means unpacking list of layers - leave only values(layers) in list
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, int(np.prod(img_shape))), # np.prod(1, 28, 28) == 1*28*28
                nn.Tanh() # result : from -1 to 1
            )

        def forward(self, z): # z == latent vector(random input vector)
            img = self.model(z) # (64, 100) --(model)--> (64, 784)
            img = img.view(img.size(0), *img_shape) # img.size(0) == N(Batch Size), (N, C, H, W) == default --> (64, 1, 28, 28)
            return img