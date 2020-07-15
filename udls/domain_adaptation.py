import torch
from . import SimpleLMDBDataset
from pathlib import Path
import librosa as li
from concurrent.futures import ProcessPoolExecutor
from os import makedirs, path
from tqdm import tqdm
import numpy as np


def dummy_load(name):
    """
    Preprocess function that takes one audio path and load it into
    chunks of 2048 samples.
    """
    x = li.load(name, 16000)[0]
    if len(x) % 2048:
        x = x[:-(len(x) % 2048)]
    x = x.reshape(-1, 2048)
    return x


class DomainAdaptationDataset(torch.utils.data.Dataset):
    def __init__(self,
                 out_database_location,
                 folder_list,
                 preprocess_function=dummy_load,
                 extension="*.wav",
                 map_size=1e9):
        super().__init__()

        self.domains = []

        makedirs(out_database_location, exist_ok=True)
        self.folder_list = folder_list
        self.preprocess_function = preprocess_function
        self.extension = extension

        for folder in folder_list:
            self.domains.append(
                SimpleLMDBDataset(
                    path.join(out_database_location,
                              path.basename(path.normpath(folder))), map_size))

        # IF NO DATA INSIDE DATASET: PREPROCESS
        self.len = np.sum([len(env) for env in self.domains])

        if self.len == 0:
            self._preprocess()
            self.len = np.sum([len(env) for env in self.domains])

        if self.len == 0:
            raise Exception("No data found !")

    def _preprocess(self):
        for index_env, (folder,
                        env) in enumerate(zip(self.folder_list, self.domains)):
            files = Path(folder).rglob(self.extension)

            index = 0

            with ProcessPoolExecutor(max_workers=16) as executor:
                for output in tqdm(
                        executor.map(self.preprocess_function, files),
                        desc=f"parsing dataset for env {index_env}"):
                    if len(output):
                        for elm in output:
                            env[index] = elm
                            index += 1

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        for i, env in enumerate(self.domains):
            if index >= len(env):
                index -= len(env)
            else:
                return i, env[index]
