import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    def __init__(self,
                 diffusion_model,
                 train_dataloader,
                 n_epochs,
                 device,
                 version):

        self.model = diffusion_model
        self.train_dataloader = train_dataloader
        self.n_epochs = n_epochs
        self.device = device
        self.version = version

        self.loss_fn = nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=2e-4)

        self.writer = SummaryWriter(log_dir=f"./tb_logs/{version}")

    def train_epoch(self, epoch):
        self.model.train()
        with tqdm(enumerate(self.train_dataloader),
                  total=len(self.train_dataloader),
                  desc=f"Train Epoch {epoch}",
                  leave=False) as pbar:
            running_loss = 0.0
            for i, batch in pbar:
                batch = batch[0].to(self.device)
                noise = torch.randn(batch.shape).to(self.device)
                timesteps = torch.randint(0, self.model.n_steps, size=(batch.shape[0], )).long().to(self.device)

                noisy_imgs = self.model.add_noise(batch, noise, timesteps)
                noise_pred = self.model.reverse(noisy_imgs, timesteps)

                loss = self.loss_fn(noise_pred, noise)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                running_loss += loss.item()
                pbar.set_postfix({"loss": running_loss / (i + 1)})
                self.writer.add_scalar("loss/train_step",
                                       loss.item(),
                                       global_step=epoch * len(self.train_dataloader) + i)

    def train(self):
        try:
            for ep in range(self.n_epochs):
                self.train_epoch(ep)
        except KeyboardInterrupt:
            print("Training interrupted.")
        torch.save(self.model.state_dict(), f"./tb_logs/{self.version}/model.pt")
