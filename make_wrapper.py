import importlib
from os import path

import torch
import torch.nn as nn

import numpy as np

from src import config
from src import get_model, compute_pca

torch.set_grad_enabled(False)
config.parse_args()

NAME = config.NAME
ROOT = path.join("runs/", config.NAME)
PCA = True

config_melgan = ".".join(path.join(ROOT, "melgan", "config").split("/"))


class BufferSTFT(nn.Module):
    def __init__(self, buffer_size, hop_length):
        super().__init__()
        n_frame = (config.BUFFER_SIZE // config.HOP_LENGTH - 1)
        buffer = torch.zeros(1, 2048 + n_frame * hop_length)
        self.register_buffer("buffer", buffer)
        self.buffer_size = buffer_size

    def forward(self, x):
        self.buffer = torch.roll(self.buffer, -self.buffer_size, -1)
        self.buffer[:, -self.buffer_size:] = x
        return self.buffer


class TracedMelEncoder(nn.Module):
    def __init__(self, melencoder, buffer, hop_length, use_buffer=True):
        super().__init__()
        self.melencoder = melencoder
        self.buffer = torch.jit.script(buffer)
        self.use_buffer = use_buffer
        self.hop_length = hop_length

    def forward(self, x):
        if self.use_buffer:
            x = self.buffer(x)
        return self.melencoder(x)


class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()

        # BUILDING MELGAN #################################################
        hparams_melgan = importlib.import_module(config_melgan).config
        hparams_melgan.override(USE_CACHED_PADDING=config.USE_CACHED_PADDING)
        melgan = get_model(hparams_melgan)

        pretrained_state_dict = torch.load(path.join(ROOT, "melgan",
                                                     "melgan_state.pth"),
                                           map_location="cpu")[0]
        state_dict = melgan.state_dict()
        state_dict.update(pretrained_state_dict)
        melgan.load_state_dict(state_dict)
        ###################################################################

        melgan.eval()

        # PRETRACE MODELS #################################################
        self.mel_size = int(config.N_MEL)

        test_wav = torch.randn(1, 8192)
        test_mel = torch.randn(1, config.INPUT_SIZE, 16)

        melencoder = TracedMelEncoder(
            melgan.encoder, BufferSTFT(config.BUFFER_SIZE, config.HOP_LENGTH),
            config.HOP_LENGTH, config.USE_CACHED_PADDING)

        self.trace_melencoder = torch.jit.trace(melencoder,
                                                test_wav,
                                                check_trace=False)
        self.trace_melgan = torch.jit.trace(melgan.decoder,
                                            test_mel,
                                            check_trace=False)

        config.override(SAMPRATE=hparams_melgan.SAMPRATE,
                        N_SIGNAL=hparams_melgan.N_SIGNAL,
                        EXTRACT_LOUDNESS=hparams_melgan.EXTRACT_LOUDNESS,
                        TYPE=hparams_melgan.TYPE,
                        HOP_LENGTH=hparams_melgan.HOP_LENGTH,
                        RATIOS=hparams_melgan.RATIOS,
                        WAV_LOC=hparams_melgan.WAV_LOC,
                        LMDB_LOC=hparams_melgan.LMDB_LOC)

        self.pca = None

        self.n_spk = melgan.num_spk

    def forward(self, x):
        return x

    @torch.jit.export
    def melencode(self, x):
        return self.trace_melencoder(x)

    @torch.jit.export
    def decode(self, x, idx):
        mel = self.melencode(x)

        # idx = b
        if len(idx.shape) == 1:
            idx = nn.functional.one_hot(idx, self.n_spk)
            idx = idx.unsqueeze(-1).float()
            idx = idx.expand(idx.shape[0], idx.shape[1], mel.shape[2]).to(mel)
        # idx = b x t
        elif len(idx.shape) == 2:
            idx = nn.functional.one_hot(idx, self.n_spk)
            idx = idx.permute(0, 2, 1).to(mel)

        mel = torch.cat([mel, idx], 1)

        waveform = self.trace_melgan(mel)

        return waveform


if __name__ == "__main__":
    wrapper = Wrapper().cpu()

    name_list = [
        config.NAME,
        str(int(np.floor(config.SAMPRATE / 1000))) + "kHz",
    ]

    name = "_".join(name_list) + ".ts"
    torch.jit.script(wrapper).save(path.join(ROOT, name))
