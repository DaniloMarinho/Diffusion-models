import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from utils.data_loading import rescale_normalized_image


class Trainer:
    def __init__(self,
                 diffusion_model,
                 train_dataloader,
                 n_epochs,
                 device,
                 version,
                 args):

        self.model = diffusion_model
        self.n_steps = self.model.n_steps
        self.train_dataloader = train_dataloader
        self.n_epochs = n_epochs
        self.device = device
        self.version = version

        self.loss_fn = nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=2e-4)

        self.writer = SummaryWriter(log_dir=f"./tb_logs/{version}")
        self.writer.add_text("config", str(args), global_step=0)

    def train_epoch(self, epoch):
        self.model.train()
        with tqdm(enumerate(self.train_dataloader),
                  total=len(self.train_dataloader),
                  desc=f"Train Epoch {epoch}",
                  leave=False) as pbar:
            running_loss = 0.0
            for i, batch in pbar:
                batch = batch[0].to(self.device)
                bs, c, h, w = batch.shape
                noise = torch.randn(batch.shape).to(self.device)
                timesteps = torch.randint(0, self.n_steps, size=(batch.shape[0], )).long().to(self.device)

                noisy_imgs = self.model.add_noise(batch, noise, timesteps)
                noise_pred = self.model.reverse(noisy_imgs, timesteps)

                loss = self.loss_fn(noise_pred, noise)

                # UNet sanity check
                #pred = self.model.network(batch, timesteps)
                #loss = self.loss_fn(pred, batch)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                running_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
                self.writer.add_scalar("loss/train_step",
                                       loss.item(),
                                       global_step=epoch * len(self.train_dataloader) + i)

                if (i + 1) % 1000 == 0 or i == 0:
                    self.log_generations(epoch * len(self.train_dataloader) + i, c, h, w)

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
                        self.writer.add_figure("denoising/{:02}".format(j),
                                               fig, global_step=epoch * len(self.train_dataloader) + i)
                        plt.close(fig)

    def train(self):
        try:
            for ep in range(self.n_epochs):
                self.train_epoch(ep)
                #self.log_generations(ep, 3, 224, 224)
        except KeyboardInterrupt:
            print("Training interrupted.")
        torch.save(self.model.state_dict(), f"./tb_logs/{self.version}/model.pt")

    def log_generations(self, global_step, c, h, w):
        cur = self.model.sample(8, c, h, w, verbose=False).cpu().detach().numpy()
        n_samples = cur.shape[0]
        for j in range(n_samples):
            fig, ax = plt.subplots()
            img = cur[j].transpose(1, 2, 0)
            img = rescale_normalized_image(img)
            ax.imshow(np.clip(img, 0., 1.), vmin=0, vmax=1)
            ax.set_axis_off()
            fig.tight_layout()
            self.writer.add_figure("training/{:02}".format(j), fig, global_step=global_step)
            plt.close(fig)
