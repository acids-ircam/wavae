from scipy.signal import kaiserord, firwin
import numpy as np

import torch
import torch.nn as nn

from src.cached_padding import CachedConv1d


def kaiser_filter(wc, atten, N=None):
    """
    Computes a kaiser lowpass filter

    Parameters
    ----------

    wc: float
        Angular frequency
    
    atten: float
        Attenuation (dB, positive)
    """
    N_, beta = kaiserord(atten, wc / np.pi)
    N_ = 2 * (N_ // 2) + 1
    N = N if N is not None else N_
    h = firwin(N, wc, window=('kaiser', beta), scale=False, nyq=np.pi)
    return h


class Resampling(nn.Module):
    def __init__(self, target_sr, source_sr):
        super().__init__()
        ratio = target_sr // source_sr
        assert int(ratio) == ratio

        wc = np.pi / ratio
        filt = kaiser_filter(wc, 140)
        filt = torch.from_numpy(filt).float()

        self.downsample = CachedConv1d(
            1,
            1,
            len(filt),
            stride=ratio,
            padding=len(filt) // 2,
            cache=True,
        )

        self.downsample.conv.weight.data.copy_(filt.reshape(1, 1, -1))
        self.downsample.conv.bias.data.zero_()

        pad = len(filt) % ratio

        filt = nn.functional.pad(filt, (pad, 0))
        filt = filt.reshape(-1, ratio).permute(1, 0)  # ratio  x T

        pad = (filt.shape[-1] + 1) % 2
        filt = nn.functional.pad(filt, (pad, 0)).unsqueeze(1)

        self.upsample = CachedConv1d(
            1,
            2,
            filt.shape[-1],
            stride=1,
            padding=filt.shape[-1] // 2,
            cache=True,
        )

        self.upsample.conv.weight.data.copy_(filt)
        self.upsample.conv.bias.data.zero_()

        self.ratio = ratio

    @torch.jit.export
    def from_target_sampling_rate(self, x):
        return self.downsample(x)

    @torch.jit.export
    def to_target_sampling_rate(self, x):
        x = self.upsample(x)  # B x 2 x T
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1).unsqueeze(1)
        return x
