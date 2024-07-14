import torch
import torch.nn as nn
from tqdm import tqdm

import pytorch_lightning as pl

from utils.images import log_generations


class DDPM(pl.LightningModule):
    def __init__(self,
                 model,
                 n_steps,
                 transform_mean,
                 transform_std,
                 dataset,
                 lr=2e-4,
                 beta_start=1e-4,
                 beta_end=2e-2):
        super(DDPM, self).__init__()

        self.model = model

        self.n_steps = n_steps

        self.dataset = dataset
        self.transform_mean = transform_mean
        self.transform_std = transform_std
        self.lr = lr

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
        return self.model(noisy, timesteps)

    @staticmethod
    def sample(diffusion_model, logger, n_samples, n_channels, height, width, log=False):
        n_steps = diffusion_model.n_steps
        device = diffusion_model.device
        betas = diffusion_model.betas
        sqrt_alphas = diffusion_model.sqrt_alphas
        sqrt_alphas_cumprod_compl = diffusion_model.sqrt_alphas_cumprod_compl
        transform_mean = diffusion_model.transform_mean
        transform_std = diffusion_model.transform_std

        diffusion_model.eval()

        with torch.no_grad():
            cur = torch.randn(n_samples, n_channels, height, width, device=device)

            timesteps = torch.linspace(n_steps - 1, 0, n_steps, device=device).long()
            for t in tqdm(timesteps, total=len(timesteps), disable=not log):
                time = torch.ones((n_samples, ), device=device) * t
                noise_coeff = betas[t] / sqrt_alphas_cumprod_compl[t]
                cur = (1 / sqrt_alphas[t]) * (cur - noise_coeff * diffusion_model(cur, time.long()))

                if log and t.item() % 100 == 0:
                    log_generations(logger=logger,
                                    gen_imgs=cur,
                                    folder="samples",
                                    global_step=n_steps - t - 1,
                                    transform_mean=transform_mean,
                                    transform_std=transform_std)

                if (t > 0).item():
                    cur = cur + torch.sqrt(betas[t]) * torch.randn(cur.shape, device=device)

        return cur

    def training_step(self, batch, batch_idx):
        x = batch[0] if type(batch) == list else batch
        bs, c, h, w = x.shape
        noise = torch.randn(x.shape, device=self.device)
        timesteps = torch.randint(0, self.n_steps, size=(x.shape[0], ), device=self.device).long()

        noisy_imgs = self.add_noise(x, noise, timesteps)
        noise_pred = self(noisy_imgs, timesteps)

        loss = self.loss_fn(noise_pred, noise)

        self.log("loss/train", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        if batch_idx == 0:
            gen_imgs = DDPM.sample(diffusion_model=self, logger=self.logger, n_samples=4, n_channels=c, height=h, width=w)
            log_generations(logger=self.logger,
                            gen_imgs=gen_imgs,
                            folder="training",
                            global_step=self.global_step,
                            transform_mean=self.transform_mean,
                            transform_std=self.transform_std)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[0] if type(batch) == list else batch
        noise = torch.randn(x.shape, device=self.device)
        timesteps = torch.randint(0, self.n_steps, size=(x.shape[0], ), device=self.device).long()

        noisy_imgs = self.add_noise(x, noise, timesteps)
        noise_pred = self(noisy_imgs, timesteps)

        loss = self.loss_fn(noise_pred, noise)

        self.log("loss/val", loss, prog_bar=False, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if self.dataset == "lsun_churches":
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)
            return [optimizer], [lr_scheduler]
        
        return optimizer
