import argparse
import json

from pytorch_lightning.loggers import TensorBoardLogger
import torch

from modules import Unet
from diffusion import DDPM

from datamodules import datamodule_selector


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()

    # Network options
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--unet_dim", type=int, default=128)
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
                                     transform_std=args.transform_std,
                                     dataset=args.dataset)


    datamodule = datamodule_selector(dataset=args.dataset,
                                     dataset_path=args.dataset_path,
                                     train_batch_size=args.batch_size,
                                     val_batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     shuffle=args.shuffle,
                                     transform_mean=args.transform_mean,
                                     transform_std=args.transform_std)
    

    logger = TensorBoardLogger(save_dir=".", version=args.version)
    logger.experiment.add_text("hyperparameters", str(vars(args)))
    samples = DDPM.sample(diffusion_model=ddpm,
                          logger=logger,
                          n_samples=args.n_samples,
                          n_channels=datamodule.n_channels,
                          height=datamodule.size,
                          width=datamodule.size,
                          log=True)

    print("Sampling finished.")
