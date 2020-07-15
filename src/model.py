import torch
import torch.nn as nn
from . import Generator, Discriminator, MelEncoder, TopVAE, config, Classifier


class Vanilla(nn.Module):
    def __init__(self, sampling_rate, hop, ratios, input_size, channels,
                 kernel, use_cached_padding, extract_loudness):
        super().__init__()
        self.melencoder = MelEncoder(sampling_rate=sampling_rate,
                                     hop=hop,
                                     input_size=input_size,
                                     center=False)
        self.topvae = TopVAE(channels=channels,
                             kernel=kernel,
                             ratios=ratios,
                             use_cached_padding=use_cached_padding,
                             extract_loudness=extract_loudness)

        if extract_loudness:
            self.classifier = Classifier()

    def forward(self, x, loudness=None):
        S = self.melencoder(x)
        y, mean_y, logvar_y, mean_z, logvar_z = self.topvae(S, loudness)
        return y, mean_y, logvar_y, mean_z, logvar_z


class melGAN(nn.Module):
    def __init__(self, sampling_rate, hop, ratios, input_size, ngf, n_res_g,
                 use_cached_padding):
        super().__init__()
        self.encoder = MelEncoder(sampling_rate=sampling_rate,
                                  hop=hop,
                                  input_size=input_size,
                                  center=False)
        self.decoder = Generator(input_size=input_size,
                                 ngf=ngf,
                                 n_residual_layers=n_res_g,
                                 ratios=ratios,
                                 use_cached_padding=use_cached_padding)

    def forward(self, x, mel_encoded=False):
        if mel_encoded:
            mel = x
        else:
            mel = self.encoder(x)

        y = self.decoder(mel)
        return y


def get_model(config=config):
    if config.TYPE == "melgan":
        return melGAN(sampling_rate=config.SAMPRATE,
                      hop=config.HOP_LENGTH,
                      ratios=config.RATIOS,
                      input_size=config.INPUT_SIZE,
                      ngf=config.NGF,
                      n_res_g=config.N_RES_G,
                      use_cached_padding=config.USE_CACHED_PADDING)

    elif config.TYPE == "vanilla":
        return Vanilla(sampling_rate=config.SAMPRATE,
                       hop=config.HOP_LENGTH,
                       ratios=config.RATIOS,
                       input_size=config.INPUT_SIZE,
                       channels=config.CHANNELS,
                       kernel=config.KERNEL,
                       use_cached_padding=config.USE_CACHED_PADDING,
                       extract_loudness=config.EXTRACT_LOUDNESS)
    else:
        raise Exception(f"Model type {config.TYPE} not understood")
