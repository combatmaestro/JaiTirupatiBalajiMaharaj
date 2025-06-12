import torch
from torch import nn
from models.encoders import psp_encoders

class pSp(nn.Module):
    def __init__(self, opts):
        super(pSp, self).__init__()
        self.opts = opts
        self.encoder = psp_encoders.Encoder4Editing(50, 'ir_se')
        self.latent_avg = None

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)
        if 'latent_avg' in state_dict:
            self.latent_avg = state_dict['latent_avg']

    def forward(self, x):
        codes = self.encoder(x)
        if self.latent_avg is not None:
            codes += self.latent_avg.repeat(codes.shape[0], 1, 1)
        return codes
