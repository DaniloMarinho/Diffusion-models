import argparse
import json
import os
from typing import Union, List

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from modules.tiny_unet import MyTinyUNet
from modules.lucidrains_unet import Unet
from diffusion.ddpm import DDPM

from datamodules import datamodule_selector


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()

    # Network options
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--unet_dim", type=int, default=64)
    parser.add_argument("--unet_dim_mults", type=int, nargs="+", default=[1, 1, 2, 2, 4, 4])

    # DataModule options
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar10", "lsun_churches"])
    parser.add_argument("--dataset_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--transform_mean", type=float, nargs="+", default=[0.5, 0.5, 0.5])
    parser.add_argument("--transform_std", type=float, nargs="+", default=[0.5, 0.5, 0.5])

    # Training options
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--devices", type=int, default=1)

    # Logging options
    parser.add_argument("--version", "-v", type=str, required=True)

    args = parser.parse_args()


    network = Unet(args.unet_dim, dim_mults=args.unet_dim_mults)
    ddpm = DDPM(model=network,
                n_steps=args.n_steps,
                transform_mean=args.transform_mean,
                transform_std=args.transform_std,
                lr=args.lr)


    datamodule = datamodule_selector(dataset=args.dataset,
                                     dataset_path=args.dataset_path,
                                     train_batch_size=args.batch_size,
                                     val_batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     shuffle=args.shuffle,
                                     transform_mean=args.transform_mean,
                                     transform_std=args.transform_std)
    

    logger = TensorBoardLogger(save_dir=".",
                               version=args.version)
    ckpt_callback = ModelCheckpoint(dirpath=f"lightning_logs/{args.version}")
    lr_monitor = LearningRateMonitor()
    trainer = pl.Trainer(logger=logger,
                         accelerator="auto",
                         devices=args.devices,
                         strategy="ddp",
                         max_epochs=args.n_epochs,
                         limit_val_batches=0.0,
                         callbacks=[ckpt_callback, lr_monitor])

    trainer.fit(ddpm, datamodule)
    

    with open(f"lightning_logs/{args.version}/config.json", "w") as f:
        json.dump(vars(args), f)

