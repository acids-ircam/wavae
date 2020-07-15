import torch
import torch.nn as nn
from . import config


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        conv = []
        for i in range(len(config.CLASSIFIER_CHANNELS) - 1):
            conv.append(
                nn.Conv1d(config.CLASSIFIER_CHANNELS[i],
                          config.CLASSIFIER_CHANNELS[i + 1],
                          5,
                          padding=5 // 2))
            if i != len(config.CLASSIFIER_CHANNELS) - 2:
                conv.append(nn.ReLU())
                conv.append(nn.BatchNorm1d(config.CLASSIFIER_CHANNELS[i + 1]))

        lin = []
        for i in range(len(config.CLASSIFIER_LIN_SIZE) - 1):
            lin.append(
                nn.Linear(config.CLASSIFIER_LIN_SIZE[i],
                          config.CLASSIFIER_LIN_SIZE[i + 1]))
            if i != len(config.CLASSIFIER_LIN_SIZE) - 2:
                lin.append(nn.ReLU())

        self.gradient_reversal = GradientReverse.apply
        self.conv = nn.Sequential(*conv)
        self.lin = nn.Sequential(*lin)

    def forward(self, z, lam=1):
        bs = z.shape[0]
        z = self.gradient_reversal(z, lam)
        z = self.conv(z)
        z = z.permute(0, 2, 1).reshape(-1, config.CLASSIFIER_LIN_SIZE[0])
        z = self.lin(z)
        z = z.reshape(bs, -1, config.CLASSIFIER_LIN_SIZE[-1]).permute(0, 2, 1)
        mean, logvar = torch.split(z, 1, 1)
        return mean, logvar