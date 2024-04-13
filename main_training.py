import argparse
import json
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from networks.tiny_unet import MyTinyUNet
from networks.lucidrains_unet import Unet
from diffusion.ddpm import DDPM
from datamodules.lsun_datamodule import LSUNDataModule

from datamodules.datamodule_selector import datamodule_selector


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()

    # Network options
    parser.add_argument("--n_steps", type=int, default=1000)

    # DataModule options
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "lsun_churches"])
    parser.add_argument("--dataset_path", type=str, required=True)

    parser.add_argument("--batch_size", "-bs", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--shuffle", type=bool, default=True)

    # Trainer options
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--devices", type=int, default=1)

    # Logging options
    parser.add_argument("--version", "-v", type=str, required=True)

    args = parser.parse_args()




    network = Unet(32, dim_mults=(1, 2, 4, 8, 16, 16))
    ddpm = DDPM(network, args.n_steps)



    datamodule = datamodule_selector(dataset=args.dataset,
                                     dataset_path=args.dataset_path,
                                     train_batch_size=args.batch_size,
                                     val_batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     shuffle=args.shuffle)
    logger = TensorBoardLogger(save_dir=".",
                               version=args.version)
    ckpt_callback = ModelCheckpoint(dirpath=f"lightning_logs/{args.version}")
    trainer = pl.Trainer(logger=logger,
                         accelerator="gpu",
                         devices=args.devices,
                         strategy="ddp",
                         max_epochs=args.n_epochs,
                         limit_val_batches=0.0,
                         callbacks=[ckpt_callback])

    trainer.fit(ddpm, datamodule)

    with open(f"lightning_logs/{args.version}/config.json", "w") as f:
        json.dump(vars(args), f)

