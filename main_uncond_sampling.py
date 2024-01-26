import argparse
import os
import json

import torch.cuda
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from models.tiny_unet import MyTinyUNet
from diffusion.ddpm import DDPM

from utils.data_loading import load_dataset

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()

    parser.add_argument("--diff_model", "-dif", default="DDPM", choices=["DDPM"])
    parser.add_argument("--n_steps", "-ns", type=int, default=1000)
    parser.add_argument("--network", "-net", default="tiny", choices=["tiny"])

    parser.add_argument("--n_samples", "-nsp", type=int, default=10)

    parser.add_argument("--weights_path", "-w", type=str, required=True)

    args = parser.parse_args()

    # Model selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    network = MyTinyUNet()
    ddpm = DDPM(network, args.n_steps, device)
    state_dict = torch.load(os.path.join(args.weights_path, "model.pt"))
    ddpm.load_state_dict(state_dict)

    f = open(os.path.join(args.weights_path, "config.json"))
    config = json.load(f)
    c, h, w = None, None, None
    if config["dataset"] == "MNIST":
        c = 1
        h = w = 32

    writer = SummaryWriter(log_dir=args.weights_path)

    ddpm.sample(10, c, h, w, writer=writer)

    writer.flush()
    writer.close()
