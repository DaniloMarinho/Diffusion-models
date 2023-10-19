import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm


class DDPM(nn.Module):
    def __init__(self,
                 network,
                 n_steps,
                 device,
                 beta_start=1e-4,
                 beta_end=2e-2):
        super(DDPM, self).__init__()

        self.network = network.to(device)

        self.n_steps = n_steps

        self.betas = torch.linspace(beta_start, beta_end, n_steps).to(device)

        self.alphas = 1 - self.betas
        self.sqrt_alphas = torch.sqrt(self.alphas)

        self.alphas_cumprod = torch.cumprod(self.alphas, -1)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_alphas_cumprod_compl = torch.sqrt(1 - self.alphas_cumprod)

    def add_noise(self, input, noise, timesteps):
        a = self.sqrt_alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
        b = self.sqrt_alphas_cumprod_compl[timesteps].reshape(-1, 1, 1, 1)
        return a * input + b * noise

    def reverse(self, noisy, timesteps):
        return self.network(noisy, timesteps)

    def sample(self, n_samples, n_channels, h, w, writer=None):
        cur = torch.randn(n_samples, n_channels, h, w).to(self.device)

        timesteps = torch.linspace(self.n_steps - 1, 0, self.n_steps).long().to(self.device)
        for t in tqdm(timesteps, total=len(timesteps)):
            time = torch.ones((n_samples, )) * t
            noise_coeff = self.betas[t] / (self.sqrt_alphas_cumprod_compl[t] * self.sqrt_alphas[t])
            cur = (1 / self.sqrt_alphas[t]) * (cur - noise_coeff * self.network(cur, time.long()))
            cur += torch.sqrt(self.betas[t]) * torch.randn(cur.shape)

            if writer is not None:
                fig, ax = plt.subplots()
                ax.imshow(cur[0, 0].cpu().detach().numpy(), cmap="gray")
                writer.add_figure("generation", fig, global_step=self.n_steps - t - 1)

        return cur
