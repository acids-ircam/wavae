import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from effortless_config import Config

import numpy as np

from src import config
from src import get_model, Discriminator, preprocess
from src import train_step_melgan, train_step_vanilla
from src import Loader, get_flattening_function, gaussian_cdf

from tqdm import tqdm
from os import path

config.parse_args()

# PREPARE DATA
dataset = Loader(config.AUGMENT)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=config.BATCH,
                                         shuffle=True,
                                         drop_last=True)

# PREPARE MODELS
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MELGAN TRAINING
if config.TYPE == "melgan":
    gen = get_model()
    dis = Discriminator()

    if config.CKPT is not None:
        ckptgen, ckptdis = torch.load(config.CKPT, map_location="cpu")
        gen.load_state_dict(ckptgen)
        dis.load_state_dict(ckptdis)

    gen = gen.to(device)
    dis = dis.to(device)

    # PREPARE OPTIMIZERS
    opt_gen = torch.optim.Adam(gen.parameters(), lr=config.LR, betas=[.5, .9])
    opt_dis = torch.optim.Adam(dis.parameters(), lr=config.LR, betas=[.5, .9])

    model = gen, dis
    opt = opt_gen, opt_dis

# VANILLA VAE TRAINING
if config.TYPE == "vanilla":
    model = get_model()
    if config.CKPT is not None:
        ckpt = torch.load(config.CKPT, map_location="cpu")
        model.load_state_dict(ckpt)
    model = model.to(device)

    # PREPARE OPTIMIZER
    opt = torch.optim.Adam(model.parameters(), lr=config.LR)

ROOT = path.join(config.PATH_PREPEND, config.NAME, config.TYPE)
writer = SummaryWriter(ROOT, flush_secs=20)

with open(path.join(ROOT, "config.py"), "w") as config_out:
    config_out.write("from effortless_config import Config\n")
    config_out.write(str(config))

# POST LOADING PROCESSING
with torch.no_grad():
    if config.TYPE == "vanilla" and config.EXTRACT_LOUDNESS:
        try:
            print("flatten loudness found, loading")
            weights, means, stds = np.load(path.join(ROOT, "flatten.npy"))
            flattening_function = gaussian_cdf(weights, means, stds)
        except:
            loudness = []
            for sample, loud_ in tqdm(dataloader, desc="parsing loudness"):
                loudness.append(loud_.reshape(-1))
            loudness = torch.cat(loudness, 0).unsqueeze(1).numpy()
            loudness = loudness[:1000000]
            print("flattening dataset loudness...")
            weights, means, stds = get_flattening_function(loudness)
            np.save(path.join(ROOT, "flatten.npy"), [weights, means, stds])
            flattening_function = gaussian_cdf(weights, means, stds)

    else:
        flattening_function = None

print("Start training !")

# TRAINING PROCESS
step = 0
for e in range(config.EPOCH):
    for batch in tqdm(dataloader):
        if config.TYPE == "vanilla":
            train_step_vanilla(model,
                               opt,
                               batch,
                               writer,
                               ROOT,
                               step,
                               device,
                               flattening=flattening_function)

        elif config.TYPE == "melgan":
            train_step_melgan(model, opt, batch, writer, ROOT, step, device)

        step += 1
