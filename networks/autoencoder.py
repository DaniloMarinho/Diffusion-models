import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = None
