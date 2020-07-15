import torch
from . import config
import torch.nn.functional as F
from os import path
import matplotlib.pyplot as plt
import numpy as np


def train_step_melgan(model, opt, data, writer, ROOT, step, device):
    gen, dis = model
    opt_gen, opt_dis = opt

    data = data.unsqueeze(1).to(device)

    y = gen(data)

    # TRAIN DISCRIMINATOR
    D_fake = dis(y.detach())
    D_real = dis(data)

    loss_D = 0

    for scale in D_fake:
        loss_D += torch.relu(1 + scale[-1]).mean()
    for scale in D_real:
        loss_D += torch.relu(1 - scale[-1]).mean()

    opt_dis.zero_grad()
    loss_D.backward()
    opt_dis.step()

    # TRAIN GENERATOR
    D_fake = dis(y)

    loss_G = 0
    for scale in D_fake:
        loss_G += -scale[-1].mean()

    loss_feat = 0
    feat_weights = 4.0 / (config.N_LAYER_D + 1)
    D_weights = 1.0 / config.NUM_D
    wt = D_weights * feat_weights
    for i in range(config.NUM_D):
        for j in range(len(D_fake[i]) - 1):
            loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())

    loss_complete = loss_G + 10 * loss_feat

    opt_gen.zero_grad()
    loss_complete.backward()
    opt_gen.step()

    writer.add_scalar("loss discriminator", loss_D, step)
    writer.add_scalar("loss adversarial", loss_G, step)
    writer.add_scalar("loss features", loss_feat, step)

    if step % config.BACKUP == 0:
        backup_name = path.join(ROOT, f"melgan_state.pth")
        states = [gen.state_dict(), dis.state_dict()]
        torch.save(states, backup_name)

    if step % config.EVAL == 0:
        writer.add_audio("original", data.reshape(-1), step, config.SAMPRATE)
        writer.add_audio("generated", y.reshape(-1), step, config.SAMPRATE)


def train_step_vanilla(model,
                       opt,
                       data,
                       writer,
                       ROOT,
                       step,
                       device,
                       flattening=None):
    if config.EXTRACT_LOUDNESS:
        sample, loudness = data
        sample = sample.to(device)
        loudness = loudness.to(device)
        fl = loudness.cpu().detach().numpy().reshape(-1)
        fl = flattening(fl)
        fl = torch.from_numpy(fl).float().to(loudness.device)
    else:
        sample = data[0].to(device)
        loudness = None

    with torch.no_grad():
        S = model.melencoder(sample)

    # COMPUTE AUTOENCODER REC AND REG LOSSES
    out = model.topvae.loss(S, loudness)
    y, mean_y, logvar_y, mean_z, logvar_z, loss_rec, loss_reg = out
    loss = loss_rec + .1 * loss_reg

    # COMPUTE DOMAIN ADAPTATION LOSS
    if config.EXTRACT_LOUDNESS:
        z = torch.randn_like(mean_z) * torch.exp(logvar_z) + mean_z
        mean_loudness, logvar_loudness = model.classifier(
            z, 1 - np.exp(-step / 100000))
        mean_loudness = torch.sigmoid(mean_loudness).reshape(-1)
        logvar_loudness = torch.clamp(logvar_loudness, -10, 0).reshape(-1)

        loss_da = torch.mean(logvar_loudness + (mean_loudness - fl)**2 *
                             torch.exp(-logvar_loudness))
        loss += loss_da

    opt.zero_grad()
    loss.backward()
    opt.step()

    writer.add_scalar("loss_rec", loss_rec, step)
    writer.add_scalar("loss_reg", loss_reg, step)

    if config.EXTRACT_LOUDNESS:
        writer.add_scalar("loss_da", loss_da, step)
        writer.add_scalar("lambda da", 1 - np.exp(-step / 100000), step)

    if step % config.BACKUP == 0:
        backup_name = path.join(ROOT, f"vanilla_state.pth")
        states = model.state_dict()
        torch.save(states, backup_name)

    if step % config.EVAL == 0:
        writer.add_histogram("mean_y", mean_y.reshape(-1), step)
        writer.add_histogram("logvar_y", logvar_y.reshape(-1), step)
        writer.add_histogram("mean_z", mean_z.reshape(-1), step)
        writer.add_histogram("logvar_z", logvar_z.reshape(-1), step)

        if config.EXTRACT_LOUDNESS:
            writer.add_histogram("mean_loudness", mean_loudness.reshape(-1),
                                 step)
            writer.add_histogram("logvar_loudness",
                                 logvar_loudness.reshape(-1), step)
            writer.add_histogram("flattened_loudness", fl.reshape(-1), step)

        ori = S.detach().cpu().numpy()
        ori = np.concatenate([o for o in ori[:4]], -1)

        rec = y.detach().cpu().numpy()
        rec = np.concatenate([r for r in rec[:4]], -1)

        img = np.concatenate([rec, ori], 0)

        plt.figure(figsize=(20, 10))
        plt.imshow(img, aspect="auto", origin="lower", cmap="magma")
        plt.axis(False)
        plt.grid(False)
        plt.tight_layout()
        writer.add_figure("reconstruction", plt.gcf(), step)
        plt.close()
