import torch
from torch import nn
from torch.nn.parallel import DataParallel
import numpy as np


def get_discriminator_A(cfg, device):
    if cfg.discriminator_A == "no_norm_5_mask_aux":
        return DataParallel(NoNormMaskAuxDiscriminator(n_layers=5, stride=1)).to(device)
    else:
        raise "Unknown Ddiscriminator A"

def get_discriminator_P(cfg, device):
    if cfg.discriminator_P == "no_norm_4_mask_aux":
        return DataParallel(NoNormMaskAuxDiscriminator(n_layers=4, stride=1)).to(device)
    else:
        raise "Unknown Ddiscriminator P"

class NoNormMaskAuxDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=32, n_layers=3, stride=2):
        super(NoNormMaskAuxDiscriminator, self).__init__()
        kw = 3
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc + 1, ndf, kernel_size=kw, stride=stride, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=stride, padding=padw),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 4)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        # sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.last = nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        self.last_aux = nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)

        self.model = nn.Sequential(*sequence)

    def forward(self, input, mask=None):
        feat = self.model(torch.cat([input, mask], dim=1))
        # feat = self.model(input)
        return self.last(feat), self.last_aux(feat)
