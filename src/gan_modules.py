import torch.nn as nn
import torch.nn.functional as F
import torch
from librosa.filters import mel as librosa_mel_fn
from torch.nn.utils import weight_norm
import numpy as np

from . import config, CachedConvTranspose1d, CachedConv1d, cache_pad


def weights_init(m):
    classname = m.__class__.__name__
    if classname == "CachedConv1d" or classname == "CachedConvTranspose1d":
        m.conv.weight.data.normal_(0.0, 0.02)
    elif classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_cached_padding=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            CachedConv1d(dim, dim, 3, 1, dilation, dilation,
                         use_cached_padding, "reflect", True),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)
        self.dilation = dilation
        self.use_cached_padding = use_cached_padding
        self.residual_padding = cache_pad(dilation,
                                          dim,
                                          cache=use_cached_padding,
                                          crop=True)

    def forward(self, x):
        blockout = self.block(x)
        shortcut = self.shortcut(x)
        if self.use_cached_padding:
            shortcut = self.residual_padding(shortcut)
        return blockout + shortcut


class Generator(nn.Module):
    def __init__(self,
                 input_size=config.INPUT_SIZE,
                 ngf=config.NGF,
                 n_residual_layers=config.N_RES_G,
                 ratios=config.RATIOS,
                 use_cached_padding=config.USE_CACHED_PADDING):

        super().__init__()
        self.hop_length = np.prod(ratios)
        mult = int(2**len(ratios))

        model = [
            CachedConv1d(input_size,
                         mult * ngf,
                         7,
                         1,
                         3,
                         cache=use_cached_padding,
                         pad_mode="reflect",
                         weight_norm=True)
        ]

        # Upsample to raw audio scale
        for i, r in enumerate(ratios):
            model += [
                nn.LeakyReLU(0.2),
                CachedConvTranspose1d(mult * ngf,
                                      mult * ngf // 2,
                                      r * 2,
                                      r,
                                      cache=use_cached_padding,
                                      weight_norm=True)
            ]

            for j in range(n_residual_layers):
                model += [
                    ResnetBlock(mult * ngf // 2,
                                dilation=3**j,
                                use_cached_padding=use_cached_padding)
                ]

            mult //= 2

        model += [
            nn.LeakyReLU(0.2),
            CachedConv1d(ngf,
                         1,
                         7,
                         1,
                         3,
                         cache=use_cached_padding,
                         pad_mode="reflect",
                         weight_norm=True),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x):
        x = self.model(x)
        return x


class NLayerDiscriminator(nn.Module):
    def __init__(self, ndf, n_layers, downsampling_factor):
        super().__init__()
        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(1, ndf, kernel_size=15),
            nn.LeakyReLU(0.2, True),
        )

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)

            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                ),
                nn.LeakyReLU(0.2, True),
            )

        nf = min(nf * 2, 1024)
        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
        )

        model["layer_%d" % (n_layers + 2)] = WNConv1d(nf,
                                                      1,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1)

        self.model = model

    def forward(self, x):
        results = []
        for key, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results


class Discriminator(nn.Module):
    def __init__(self,
                 num_D=config.NUM_D,
                 ndf=config.NDF,
                 n_layers=config.N_LAYER_D,
                 downsampling_factor=config.DOWNSAMP_D):
        super().__init__()
        self.model = nn.ModuleDict()
        for i in range(num_D):
            self.model[f"disc_{i}"] = NLayerDiscriminator(
                ndf, n_layers, downsampling_factor)

        self.downsample = nn.AvgPool1d(4,
                                       stride=2,
                                       padding=1,
                                       count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x):
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))
            x = self.downsample(x)
        return results
