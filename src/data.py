from udls import SimpleDataset
import librosa as li
import torch
import torch.nn as nn

import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.special import erf

from tqdm import tqdm

from . import config


class Loader(torch.utils.data.Dataset):
    def __init__(self, cat, config=config):
        super().__init__()
        if config.WAV_LOC is not None:
            wav_loc = config.WAV_LOC.split(",")
        else:
            wav_loc = None
        self.dataset = SimpleDataset(config.LMDB_LOC,
                                     folder_list=config.WAV_LOC,
                                     file_list=config.FILE_LIST,
                                     preprocess_function=preprocess,
                                     map_size=1e11)
        self.cat = cat

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if config.TYPE == "vanilla":
            sample = []
            loudness = []
            for i in range(self.cat):
                s, l = self.dataset[(idx + i) % self.__len__()]
                sample.append(torch.from_numpy(s).float())
                loudness.append(torch.from_numpy(l).float())
            sample = torch.cat(sample, -1)
            loudness = torch.cat(loudness, -1)
            return sample, loudness
        else:
            return self.dataset[idx]
