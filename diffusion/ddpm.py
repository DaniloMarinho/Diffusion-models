import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.data_loading import rescale_normalized_image


class DDPM(nn.Module):
    def __init__(self,
                 network,
                 n_steps,
                 beta_start=1e-4,
                 beta_end=2e-2):
        super(DDPM, self).__init__()

        self.network = network

        self.n_steps = n_steps

        self.betas = torch.linspace(beta_start, beta_end, n_steps)

        self.alphas = 1 - self.betas
        self.sqrt_alphas = torch.sqrt(self.alphas)

        self.alphas_cumprod = torch.cumprod(self.alphas, -1)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_alphas_cumprod_compl = torch.sqrt(1 - self.alphas_cumprod)

    def to_device(self, rank):
        self.betas = self.betas.to(rank)

        self.alphas = self.alphas.to(rank)
        self.sqrt_alphas = self.sqrt_alphas.to(rank)

        self.alphas_cumprod = self.alphas_cumprod.to(rank)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(rank)
        self.sqrt_alphas_cumprod_compl = self.sqrt_alphas_cumprod_compl.to(rank)

    def add_noise(self, input, noise, timesteps):
        a = self.sqrt_alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
        b = self.sqrt_alphas_cumprod_compl[timesteps].reshape(-1, 1, 1, 1)
        return a * input + b * noise

    def reverse(self, noisy, timesteps):
        return self.network(noisy, timesteps)

    def sample(self, rank, n_samples, n_channels, h, w, writer=None, verbose=True):
        with torch.no_grad():
            cur = torch.randn(n_samples, n_channels, h, w).to(rank)

            timesteps = torch.linspace(self.n_steps - 1, 0, self.n_steps).long().to(rank)
            for t in tqdm(timesteps, total=len(timesteps), disable=not verbose):
                time = torch.ones((n_samples, ), device=rank) * t
                noise_coeff = self.betas[t] / (self.sqrt_alphas_cumprod_compl[t] * self.sqrt_alphas[t])
                cur = (1 / self.sqrt_alphas[t]) * (cur - noise_coeff * self.network(cur, time.long()))
                cur = cur + torch.sqrt(self.betas[t]) * torch.randn(cur.shape, device=rank)

                if writer is not None and (t + 1) % 50 == 0:
                    for i in range(0, n_samples):
                        fig, ax = plt.subplots()
                        img = cur[i].cpu().detach().numpy().transpose(1, 2, 0)
                        img = rescale_normalized_image(img)
                        ax.imshow(img)
                        ax.set_axis_off()
                        fig.tight_layout()
                        writer.add_figure("generation/{:03}".format(i), fig, global_step=self.n_steps - t - 1)
                        plt.close(fig)

        return cur
