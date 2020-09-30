import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from effortless_config import Config

import numpy as np

from src import config
from src import get_model, Discriminator
from src import train_step_melgan, train_step_vanilla
from src import Loader

from tqdm import tqdm
from os import path

config.parse_args()
# config.override(WAV_LOC="/Users/caillon/Desktop/")
# config.override(EVAL=10)

# PREPARE DATA
dataset = Loader(config)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config.BATCH,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

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
                               flattening=None)

        elif config.TYPE == "melgan":
            train_step_melgan(model, opt, batch, writer, ROOT, step, device)

        step += 1
