import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks.text_encoder import Transformer, LSTM_Text_Encoder


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if m.weight.requires_grad:
            m.weight.data.normal_(std=0.02)
        if m.bias is not None and m.bias.requires_grad:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d) and m.affine:
        if m.weight.requires_grad:
            m.weight.data.normal_(1, 0.02)
        if m.bias.requires_grad:
            m.bias.data.fill_(0)


class ResidualBlock(nn.Module):
    def __init__(self, ndim):
        super(ResidualBlock, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(ndim, ndim, 3, padding=1, bias=False),
            nn.BatchNorm2d(ndim),
            nn.ReLU(inplace=True),
            nn.Conv2d(ndim, ndim, 3, padding=1, bias=False),
            nn.BatchNorm2d(ndim)
        )

    def forward(self, x):
        return x + self.encoder(x)


class Generator(nn.Module):
    def __init__(self, cfg, text_encoder):
        super(Generator, self).__init__()
        self.cfg = cfg
        self.t_encoder = self.cfg.t_encoder
        self.embedding_dim = self.cfg.embedding_dim
        self.w_dim = self.cfg.w_dim

        self.text_encoder = text_encoder

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # residual blocks
        self.residual_blocks = nn.Sequential(
            nn.Conv2d(512 + 128, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512)
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )

        # conditioning augmentation
        self.mu = nn.Sequential(
            nn.Linear(self.w_dim, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.log_sigma = nn.Sequential(
            nn.Linear(self.w_dim, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.apply(init_weights)

    def forward(self, img, txt, txt_len):
        # image encoder
        e = self.encoder(img)

        # text encoder
        h, cond, mask = self.text_encoder(txt, txt_len)

        z_mean = self.mu(cond)
        z_log_stddev = self.log_sigma(cond)
        z = torch.randn(cond.size(0), 128, device=h.device)
        cond = z_mean + z_log_stddev.exp() * z

        # residual blocks
        cond = cond.unsqueeze(-1).unsqueeze(-1)
        merge = self.residual_blocks(torch.cat((e, cond.repeat(1, 1, e.size(2), e.size(3))), 1))

        # decoder
        d = self.decoder(e + merge)
        d = torch.nn.functional.tanh(d)

        return d, (z_mean, z_log_stddev)


