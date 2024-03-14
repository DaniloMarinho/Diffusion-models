import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.data_loading import load_dataset, rescale_normalized_image

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def ddpm_train(rank,
               world_size,
               model,
               dataset,
               dataset_path,
               batch_size,
               n_epochs,
               version):
    setup(rank, world_size)

    writer = SummaryWriter(log_dir=f"./tb_logs/{version}")

    model.to(rank)
    model.to_device(rank)
    n_steps = model.n_steps
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=2e-4)

    train_dataset = load_dataset(dataset=dataset, dataset_path=dataset_path)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=16)

    loss_fn = nn.MSELoss().to(rank)

    try:
        for ep in range(n_epochs):
            model.train()
            train_dataloader.sampler.set_epoch(ep)
            with tqdm(enumerate(train_dataloader),
                      total=len(train_dataloader),
                      desc=f"Train Epoch {ep}",
                      leave=False,
                      disable=(rank != 0)) as pbar:
                running_loss = 0.0
                for i, batch in pbar:
                    batch = batch[0].to(rank)
                    bs, c, h, w = batch.shape
                    noise = torch.randn(batch.shape).to(rank)
                    timesteps = torch.randint(0, n_steps, size=(batch.shape[0],)).long().to(rank)

                    noisy_imgs = ddp_model.module.add_noise(batch, noise, timesteps)
                    noise_pred = ddp_model.module.reverse(noisy_imgs, timesteps)

                    loss = loss_fn(noise_pred, noise)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if rank == 0:
                        running_loss += loss.item()
                        pbar.set_postfix({"loss": loss.item()})
                        writer.add_scalar("loss/train_step",
                                               loss.item(),
                                               global_step=ep * len(train_dataloader) + i)

                        if (i + 1) % 1000 == 0 or i == 0:
                            log_generations(rank, ddp_model.module, ep * len(train_dataloader) + i, c, h, w, writer)

                            # if (i + 1) % 10 == 0:
                            for j in range(4):
                                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                                img_pred = noisy_imgs[j].cpu().detach().numpy().transpose(1, 2, 0)
                                img_pred = rescale_normalized_image(img_pred)
                                ax[0].imshow(np.clip(img_pred, 0, 1), vmin=0, vmax=1)
                                img_gt = batch[j].cpu().detach().numpy().transpose(1, 2, 0)
                                img_gt = rescale_normalized_image(img_gt)
                                ax[1].imshow(np.clip(img_gt, 0, 1), vmin=0, vmax=1)
                                img_diff = (0.5 + noise_pred - noise)[j].cpu().detach().numpy().transpose(1, 2, 0)
                                img_diff = rescale_normalized_image(img_diff)
                                ax[2].imshow(np.clip(img_diff, 0, 1), vmin=0, vmax=1)
                                ax[0].set_axis_off()
                                ax[1].set_axis_off()
                                ax[2].set_axis_off()
                                fig.tight_layout()
                                writer.add_figure("denoising/{:02}".format(j),
                                                       fig, global_step=ep * len(train_dataloader) + i)
                                plt.close(fig)
    except KeyboardInterrupt:
        print("Training interrupted.")

    writer.flush()
    writer.close()

    cleanup()

def log_generations(rank, model, global_step, c, h, w, writer):
    cur = model.sample(rank, 8, c, h, w, verbose=False).cpu().detach().numpy()
    n_samples = cur.shape[0]
    for j in range(n_samples):
        fig, ax = plt.subplots()
        img = cur[j].transpose(1, 2, 0)
        img = rescale_normalized_image(img)
        ax.imshow(np.clip(img, 0., 1.), vmin=0, vmax=1)
        ax.set_axis_off()
        fig.tight_layout()
        writer.add_figure("training/{:02}".format(j), fig, global_step=global_step)
        plt.close(fig)