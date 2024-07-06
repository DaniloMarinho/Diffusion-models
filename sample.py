import argparse
import json
import os
import ast
from typing import Union, List

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import matplotlib.pyplot as plt

from modules.lucidrains_unet import Unet
from diffusion.ddpm import DDPM

from datamodules import datamodule_selector
from utils.images import log_generations


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()

    # Network options
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--unet_dim", type=int, default=64)
    parser.add_argument("--unet_dim_mults", type=int, nargs="+", default=[1, 1, 2, 2, 4, 4])

    # Sampling options
    parser.add_argument("--n_samples", type=int, default=8)

    # DataModule options
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar10", "lsun_churches"])
    parser.add_argument("--dataset_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--transform_mean", type=float, nargs="+", default=[0.5, 0.5, 0.5])
    parser.add_argument("--transform_std", type=float, nargs="+", default=[0.5, 0.5, 0.5])

    # Logging options
    parser.add_argument("--version", "-v", type=str, required=True)

    args = parser.parse_args()


    device = torch.cuda.is_available()

    network = Unet(args.unet_dim, dim_mults=args.unet_dim_mults)
    ddpm = DDPM.load_from_checkpoint(args.ckpt_path,
                                     model=network,
                                     n_steps=args.n_steps,
                                     transform_mean=args.transform_mean,
                                     transform_std=args.transform_std)


    datamodule = datamodule_selector(dataset=args.dataset,
                                     dataset_path=args.dataset_path,
                                     train_batch_size=args.batch_size,
                                     val_batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     shuffle=args.shuffle,
                                     transform_mean=args.transform_mean,
                                     transform_std=args.transform_std)
    

    logger = TensorBoardLogger(save_dir=".", version=args.version)

    ddpm.eval()
    samples = DDPM.sample(diffusion_model=ddpm,
                          logger=logger,
                          n_samples=args.n_samples,
                          n_channels=datamodule.n_channels,
                          height=datamodule.size,
                          width=datamodule.size,
                          log=True)
    # log_generations(logger=logger,
    #                 gen_imgs=samples,
    #                 folder="samples",
    #                 global_step=0,
    #                 transform_mean=args.transform_mean,
    #                 transform_std=args.transform_std,
    #                 verbose=True)

    print("Sampling finished.")

