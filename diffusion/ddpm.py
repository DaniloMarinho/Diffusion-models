import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

import pytorch_lightning as pl

from utils.images import rescale_normalized_image


class DDPM(pl.LightningModule):
    def __init__(self,
                 network,
                 n_steps,
                 beta_start=1e-4,
                 beta_end=2e-2):
        super(DDPM, self).__init__()

        self.network = network

        self.n_steps = n_steps

        self.betas = nn.Parameter(torch.linspace(beta_start, beta_end, n_steps), requires_grad=False)

        self.alphas = 1 - self.betas
        self.sqrt_alphas = nn.Parameter(torch.sqrt(self.alphas), requires_grad=False)

        self.alphas_cumprod = torch.cumprod(self.alphas, -1)
        self.sqrt_alphas_cumprod = nn.Parameter(torch.sqrt(self.alphas_cumprod), requires_grad=False)
        self.sqrt_alphas_cumprod_compl = nn.Parameter(torch.sqrt(1 - self.alphas_cumprod), requires_grad=False)

        self.loss_fn = nn.MSELoss()

    def add_noise(self, input, noise, timesteps):
        a = self.sqrt_alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
        b = self.sqrt_alphas_cumprod_compl[timesteps].reshape(-1, 1, 1, 1)
        return a * input + b * noise

    def forward(self, noisy, timesteps):
        return self.network(noisy, timesteps)

    def sample(self, n_samples, n_channels, height, width, log=False):
        with torch.no_grad():
            cur = torch.randn(n_samples, n_channels, height, width, device=self.device)

            timesteps = torch.linspace(self.n_steps - 1, 0, self.n_steps).long().to(self.device)
            for t in tqdm(timesteps, total=len(timesteps), disable=not log):
                time = torch.ones((n_samples, ), device=self.device) * t
                noise_coeff = self.betas[t] / (self.sqrt_alphas_cumprod_compl[t] * self.sqrt_alphas[t])
                cur = (1 / self.sqrt_alphas[t]) * (cur - noise_coeff * self.network(cur, time.long()))
                cur = cur + torch.sqrt(self.betas[t]) * torch.randn(cur.shape, device=self.device)

                if log and t + 1 % 100 == 0:
                    self.log_generations(gen_imgs=cur, folder="generation", global_step=self.n_steps - t - 1)

        return cur

    def training_step(self, batch, batch_idx):
        bs, c, h, w = batch.shape
        noise = torch.randn(batch.shape).to(self.device)
        timesteps = torch.randint(0, self.n_steps, size=(batch.shape[0], )).long().to(self.device)

        noisy_imgs = self.add_noise(batch, noise, timesteps)
        noise_pred = self(noisy_imgs, timesteps)

        loss = self.loss_fn(noise_pred, noise)

        self.log("loss/train", loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        if batch_idx % 1000 == 0:
            gen_imgs = self.sample(n_samples=4, n_channels=c, height=h, width=w)
            self.log_generations(gen_imgs=gen_imgs, folder="training", global_step=self.global_step)

        return loss
    
    def validation_step(self, batch, batch_idx):
        noise = torch.randn(batch.shape).to(self.device)
        timesteps = torch.randint(0, self.n_steps, size=(batch.shape[0], )).long().to(self.device)

        noisy_imgs = self.add_noise(batch, noise, timesteps)
        noise_pred = self(noisy_imgs, timesteps)

        loss = self.loss_fn(noise_pred, noise)

        self.log("loss/val", loss, on_epoch=True, logger=True, sync_dist=True)

        return loss

    def log_generations(self, gen_imgs, folder, global_step):
        n_samples = gen_imgs.shape[0]
        for i in range(n_samples):
            fig, ax = plt.subplots()
            img = gen_imgs[i].cpu().detach().numpy().transpose(1, 2, 0)
            img = rescale_normalized_image(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ax.imshow(img)
            ax.set_axis_off()
            fig.tight_layout()
            self.logger.experiment.add_figure(f"{folder}/{i}", fig, global_step=global_step)
            plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=2e-4)
        return [optimizer]

