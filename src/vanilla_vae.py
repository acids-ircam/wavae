import torch
import torch.nn as nn
from . import config, CachedConv1d, CachedConvTranspose1d
import numpy as np


class ConvEncoder(nn.Module):
    """
    Multi Layer Convolutional Variational Encoder
    """
    def __init__(self, channels, kernel, ratios, use_cached_padding):
        super().__init__()

        self.channels = channels
        self.kernel = kernel
        self.ratios = ratios

        self.convs = []
        for i in range(len(self.ratios)):
            self.convs += [
                CachedConv1d(self.channels[i],
                             self.channels[i + 1],
                             self.kernel,
                             padding=self.kernel // 2,
                             stride=self.ratios[i],
                             cache=use_cached_padding)
            ]
            if i != len(self.ratios) - 1:
                self.convs += [nn.ReLU(), nn.BatchNorm1d(self.channels[i + 1])]

        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        x = self.convs(x)
        return x


class ConvDecoder(nn.Module):
    """
    Multi Layer Convolutional Variational Decoder
    """
    def __init__(self, channels, ratios, kernel, use_cached_padding,
                 extract_loudness):

        self.channels = channels
        self.ratios = ratios
        self.kernel = kernel

        super().__init__()
        self.channels = list(self.channels)
        self.channels[0] *= 2
        self.channels[-1] //= 2

        if extract_loudness:
            self.channels[-1] += 1

        self.convs = []

        for i in range(len(self.ratios))[::-1]:
            if self.ratios[i] == 1:
                self.convs += [
                    CachedConv1d(self.channels[i + 1],
                                 self.channels[i],
                                 self.kernel,
                                 stride=1,
                                 padding=self.kernel // 2,
                                 cache=use_cached_padding)
                ]

            else:
                self.convs += [
                    CachedConvTranspose1d(self.channels[i + 1],
                                          self.channels[i],
                                          2 * self.ratios[i],
                                          stride=self.ratios[i],
                                          cache=use_cached_padding)
                ]
            if i:
                self.convs += [nn.ReLU(), nn.BatchNorm1d(self.channels[i])]

        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        x = self.convs(x)
        return x


class TopVAE(nn.Module):
    """
    Top Variational Auto Encoder
    """
    def __init__(self, channels, kernel, ratios, use_cached_padding,
                 extract_loudness):
        super().__init__()
        self.encoder = ConvEncoder(channels, kernel, ratios,
                                   use_cached_padding)
        self.decoder = ConvDecoder(channels, ratios, kernel,
                                   use_cached_padding, extract_loudness)

        self.channels = channels

        skipped = 0
        for p in self.parameters():
            try:
                nn.init.xavier_normal_(p)
            except:
                skipped += 1

    def encode(self, x):
        out = self.encoder(x)
        mean, logvar = torch.split(out, self.channels[-1] // 2, 1)
        z = torch.randn_like(mean) * torch.exp(logvar) + mean
        return z, mean, logvar

    def decode(self, z):
        rec = self.decoder(z)
        mean, logvar = torch.split(rec, self.channels[0], 1)
        mean = torch.sigmoid(mean)
        logvar = torch.clamp(logvar, min=-10, max=0)
        y = torch.randn_like(mean) * torch.exp(logvar) + mean
        return y, mean, logvar

    def deterministic_decode(self, z):
        rec = self.decoder(z)
        mean = torch.split(rec, self.channels[0], 1)[0]
        return torch.sigmoid(mean)

    def forward(self, x, loudness):
        z, mean_z, logvar_z = self.encode(x)
        if loudness is not None:
            z = torch.cat([loudness, z], 1)
        y, mean_y, logvar_y = self.decode(z)
        return y, mean_y, logvar_y, mean_z, logvar_z

    def loss(self, x, loudness):
        y, mean_y, logvar_y, mean_z, logvar_z = self.forward(x, loudness)

        loss_rec = logvar_y + (x - mean_y)**2 * torch.exp(-logvar_y)

        loss_reg = mean_z**2 + torch.exp(logvar_z) - logvar_z - 1

        loss_rec = torch.mean(loss_rec)
        loss_reg = torch.mean(loss_reg)

        return y, mean_y, logvar_y, mean_z, logvar_z, loss_rec, loss_reg
