import argparse
import json

import torch.cuda
from torch.utils.data import DataLoader

from models.tiny_unet import MyTinyUNet
from models.unet import UNet
from diffusion.ddpm import DDPM

from utils.data_loading import load_dataset
from trainer import Trainer

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()

    parser.add_argument("--diff_model", "-dif", default="DDPM", choices=["DDPM"])
    parser.add_argument("--n_steps", "-ns", type=int, default=1000)
    parser.add_argument("--network", "-net", default="tiny", choices=["tiny"])

    parser.add_argument("--dataset", "-d", type=str)
    parser.add_argument("--dataset_path", "-dp", type=str)

    parser.add_argument("--batch_size", "-bs", type=int, default=1024)
    parser.add_argument("--n_epochs", "-ne", type=int, default=50)

    parser.add_argument("--version", "-v", type=str, required=True)

    args = parser.parse_args()

    # Model selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    network = UNet(n_channels=3)
    ddpm = DDPM(network, args.n_steps, device)

    # Data loading
    train_dataset = load_dataset(dataset=args.dataset, dataset_path=args.dataset_path)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Train
    trainer = Trainer(ddpm, train_dataloader, args.n_epochs, device, args.version, vars(args))
    trainer.train()

    with open(f"tb_logs/{args.version}/config.json", "w") as f:
        json.dump(vars(args), f)
