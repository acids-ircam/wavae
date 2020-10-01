from udls import SimpleDataset, simple_audio_preprocess
import udls.transforms
from .hparams import Config
import yaml
from os import path


def get_preprocess(config):
    def preprocess(name):
        wav = simple_audio_preprocess(
            config.SAMPRATE,
            config.N_SIGNAL + config.N_SIGNAL // 4,
        )(name)
        with open(config.INDEX_FILE, "r") as indexes:
            indexes = yaml.safe_load(indexes)

        idx = indexes[path.basename(name)]

        return zip(wav, wav.shape[0] * [idx])

    return preprocess


class SkipElse(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, data):
        x, *stuff = data
        data = [self.transform(x)]
        data.extend(stuff)
        return data


def get_transforms(config):
    transforms = SkipElse(
        udls.transforms.Compose([
            udls.transforms.Dequantize(16),
            udls.transforms.RandomApply(
                udls.transforms.RandomChoice([
                    udls.transforms.Reverb(sr=config.SAMPRATE),
                    udls.transforms.PitchShift(sr=config.SAMPRATE),
                    udls.transforms.Noise(),
                ]),
                p=.7,
            ),
            udls.transforms.RandomCrop(config.N_SIGNAL),
        ]))

    return transforms


class Loader(SimpleDataset):
    def __init__(self, config):
        super().__init__(
            out_database_location=config.LMDB_LOC,
            folder_list=config.WAV_LOC,
            preprocess_function=get_preprocess(config),
            transforms=get_transforms(config),
            extension="*.wav,*.aif",
            map_size=1e11,
            split_percent=.2,
            split_set="full",
            seed=0,
        )
