import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os


class Sampler:
    def __init__(self,
                 diffusion_model,
                 network,
                 device,
                 weights_path):
        self.model = diffusion_model.to(device)
        state_dict = torch.load(os.path.join(weights_path, "model.pt"))
        self.model.load_state_dict(state_dict)

        self.n_steps = diffusion_model.n_steps
        self.device = device

        self.betas = self.model.betas.to(device)
        self.sqrt_alphas_cumprod_compl = self.model.sqrt_alphas_cumprod_compl.to(device)
        self.sqrt_alphas = self.model.sqrt_alphas.to(device)

        self.writer = SummaryWriter(log_dir=weights_path)

    def sample(self, n_samples, n_channels, h, w):
        cur = torch.randn(n_samples, n_channels, h, w).to(self.device)

        timesteps = torch.linspace(self.n_steps - 1, 0, self.n_steps).long().to(self.device)
        for t in tqdm(timesteps, total=len(timesteps)):
            time = torch.ones((n_samples,), device=self.device) * t

            noise_coeff = self.betas[t] / (self.sqrt_alphas_cumprod_compl[t] * self.sqrt_alphas[t])

            cur = (1 / self.sqrt_alphas[t]) * (cur - noise_coeff * self.model.network(cur, time.long()))
            cur += torch.sqrt(self.betas[t]) * torch.randn(cur.shape, device=self.device)

            if t % 50 == 0:
                for i in range(n_samples):
                    fig, ax = plt.subplots()
                    ax.imshow(cur[i, 0].cpu().detach().numpy(), cmap="gray")
                    self.writer.add_figure("generation/{:03}".format(i), fig, global_step=self.n_steps - t - 1)
                    plt.close(fig)

        return cur
