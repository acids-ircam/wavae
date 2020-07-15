import torch
from . import SimpleLMDBDataset
from pathlib import Path
import librosa as li
from concurrent.futures import ProcessPoolExecutor, TimeoutError
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
    if x.shape[0]:
        return x
    else:
        return None


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        out_database_location,
        folder_list=None,
        file_list=None,
        preprocess_function=dummy_load,
        extension="*.wav,*.aif",
        map_size=1e9,
        multiprocess=True,
        split_percent=.2,
        split_set="train",
        seed=0,
    ):
        super().__init__()

        assert folder_list is not None or file_list is not None

        self.env = SimpleLMDBDataset(out_database_location, map_size)

        self.folder_list = folder_list
        self.file_list = file_list

        self.preprocess_function = preprocess_function
        self.extension = extension
        self.multiprocess = multiprocess

        makedirs(out_database_location, exist_ok=True)

        # IF NO DATA INSIDE DATASET: PREPROCESS
        self.len = len(self.env)

        if self.len == 0:
            self._preprocess()
            self.len = len(self.env)

        if self.len == 0:
            raise Exception("No data found !")

        self.index = np.arange(self.len)
        np.random.seed(seed)
        np.random.shuffle(self.index)

        if split_set == "train":
            self.len = int(np.floor((1 - split_percent) * self.len))
            self.offset = 0

        elif split_set == "test":
            self.offset = int(np.floor((1 - split_percent) * self.len))
            self.len = self.len - self.offset

        elif split_set == "full":
            self.offset = 0

    def _preprocess(self):
        extension = self.extension.split(",")
        idx = 0
        wavs = []

        # POPULATE WAV LIST
        if self.folder_list is not None:
            for f, folder in enumerate(self.folder_list.split(",")):
                print("Recursive search in {}".format(folder))
                for ext in extension:
                    wavs.extend(list(Path(folder).rglob(ext)))

        else:
            with open(self.file_list, "r") as file_list:
                wavs = file_list.read().split("\n")

        # CREATE ASYNCHRONOUS PREPROCESS TASKS
        if self.multiprocess:
            futures = []
            with ProcessPoolExecutor() as executor:
                for wav in wavs:
                    futures.append((path.basename(wav),
                                    executor.submit(self.preprocess_function,
                                                    wav)))
                loader = tqdm(futures)
                for name, f in loader:
                    loader.set_description("{}".format(name))
                    try:
                        output = f.result(timeout=60)
                    except TimeoutError:
                        output = None
                        print("Failed to preprocess {}".format(name))
                    if output is not None:
                        for o in output:
                            self.env[idx] = o
                            idx += 1
        else:
            loader = tqdm(wavs)
            for wav in loader:
                loader.set_description("{}".format(path.basename(wav)))
                output = self.preprocess_function(wav)
                if output is not None:
                    for o in output:
                        self.env[idx] = o
                        idx += 1

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.env[self.index[index + self.offset]]