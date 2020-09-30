import torch
import librosa as li
from random import random, choice, randint
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


class Transform(object):
    def __call__(self, x: torch.Tensor):
        raise NotImplementedError


class RandomApply(Transform):
    def __init__(self, transform, p=.5):
        self.transform = transform
        self.p = p

    def __call__(self, x: np.ndarray):
        if random() < self.p:
            x = self.transform(x)
        return x


class Compose(Transform):
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, x: np.ndarray):
        for elm in self.transform_list:
            x = elm(x)
        return x


class RandomChoice(Transform):
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, x: np.ndarray):
        x = choice(self.transform_list)(x)
        return x


class PitchShift(Transform):
    def __init__(self, mean=0, std=3, sr=24000):
        self.mean = mean
        self.std = std
        self.sr = sr

    def __call__(self, x: np.ndarray):
        r = self.std * (random() - .5) + self.mean
        x = li.effects.pitch_shift(x, self.sr, r, res_type="kaiser_fast")
        return x


class Reverb(Transform):
    def __init__(self, mean=30, std=30, sr=24000):
        self.mean = mean
        self.std = std
        self.sr = sr

    def __call__(self, x: np.ndarray):
        r = self.std * (random() - .5) + self.mean

        noise = 2 * np.random.rand(self.sr) - 1
        fade = np.linspace(1, 0, self.sr)
        exp = np.exp(-np.linspace(0, r, self.sr))

        impulse = noise * fade * exp * .1
        impulse[0] = 1

        shape_x_ori = len(x)

        y = fftconvolve(x, impulse, "full")

        return y[:shape_x_ori]


class Noise(Transform):
    def __init__(self, std=1e-4):
        self.std = std

    def __call__(self, x: np.ndarray):
        return x + self.std * (2 * np.random.rand(len(x)) - 1)


class RandomCrop(Transform):
    def __init__(self, n_signal):
        self.n_signal = n_signal

    def __call__(self, x: np.ndarray):
        in_point = randint(0, len(x) - self.n_signal)
        x = x[in_point:in_point + self.n_signal]
        return x


class Dequantize(Transform):
    def __init__(self, bit_depth):
        self.bit_depth = bit_depth

    def __call__(self, x: np.ndarray):
        x += np.random.rand(len(x)) / 2**self.bit_depth
        return x