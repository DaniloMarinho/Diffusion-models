import argparse

import torch.cuda
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from nets.tiny_unet import MyTinyUNet
from diff_models.ddpm import DDPM

from utils.data_loading import load_dataset
from trainer import Trainer

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()

    parser.add_argument("--diff_model", "-dif", default="DDPM", choices=["DDPM"])
    parser.add_argument("--n_steps", "-ns", type=int, default=1000)
    parser.add_argument("--network", "-net", default="tiny", choices=["tiny"])

    parser.add_argument("--n_samples", "-nsp", type=int, default=1)

    parser.add_argument("--weights", "-w", type=str, required=True)

    args = parser.parse_args()

    # Model selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    network = MyTinyUNet()
    ddpm = DDPM(network, args.n_steps)
    state_dict = torch.load(args.weights, map_location="cpu")
    ddpm.load_state_dict(state_dict)

    generated = ddpm.sample(args.n_samples, 1, 32, 32)
    generated = generated.cpu().detach().numpy()
    