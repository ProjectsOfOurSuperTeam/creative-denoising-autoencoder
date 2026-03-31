"""
Denoising Autoencoder (28x28 grayscale) — same architecture as main.ipynb.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv → BN → LeakyReLU block."""

    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DenoisingAutoencoder(nn.Module):
    """
    Convolutional Denoising Autoencoder for 28×28 grayscale images.
    Output: sigmoid, pixels in [0, 1].
    """

    def __init__(self, base_ch: int = 32, latent_dim: int = 128):
        super().__init__()
        self.base_ch = base_ch
        self.latent_dim = latent_dim

        self.enc1 = ConvBlock(1, base_ch, stride=2)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, stride=2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, stride=1)

        enc_flat = base_ch * 4 * 7 * 7
        self.fc_enc = nn.Linear(enc_flat, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, enc_flat)
        self.bn_bottleneck = nn.BatchNorm1d(latent_dim)

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2 + base_ch * 2, base_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch + base_ch, base_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.out_conv = nn.Conv2d(base_ch, 1, 1)

    def encode(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        z = self.fc_enc(s3.flatten(1))
        z = self.bn_bottleneck(z)
        return z, s1, s2

    def decode(self, z, s1, s2):
        b, c = self.base_ch * 4, 7
        h = self.fc_dec(z).view(-1, b, c, c)
        h = self.dec3(h)
        h = self.dec2(torch.cat([h, s2], dim=1))
        h = self.dec1(torch.cat([h, s1], dim=1))
        return torch.sigmoid(self.out_conv(h))

    def forward(self, x):
        z, s1, s2 = self.encode(x)
        return self.decode(z, s1, s2)
