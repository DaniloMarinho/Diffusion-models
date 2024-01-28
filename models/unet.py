from models.unet_parts import *
from utils.embedding import sinusoidal_embedding

class UNet(nn.Module):
    def __init__(self, n_channels,  n_steps=1000, time_emb_dim=100, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        self.te_in = self._make_te(time_emb_dim, n_channels)
        self.te_down1 = self._make_te(time_emb_dim, 64)
        self.te_down2 = self._make_te(time_emb_dim, 128)
        self.te_down3 = self._make_te(time_emb_dim, 256)
        self.te_down4 = self._make_te(time_emb_dim, 512)
        self.te_up1 = self._make_te(time_emb_dim, 1024)
        self.te_up2 = self._make_te(time_emb_dim, 512)
        self.te_up3 = self._make_te(time_emb_dim, 256)
        self.te_up4 = self._make_te(time_emb_dim, 128)
        self.te_out = self._make_te(time_emb_dim, 64)

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_channels))

    def forward(self, x, t):
        t = self.time_embed(t)
        n = len(x)

        x1 = self.inc(x + self.te_in(t).reshape(n, -1, 1, 1))
        x2 = self.down1(x1 + self.te_down1(t).reshape(n, -1, 1, 1))
        x3 = self.down2(x2 + self.te_down2(t).reshape(n, -1, 1, 1))
        x4 = self.down3(x3 + self.te_down3(t).reshape(n, -1, 1, 1))
        x5 = self.down4(x4 + self.te_down4(t).reshape(n, -1, 1, 1))
        x = self.up1(x5, x4, self.te_up1(t).reshape(n, -1, 1, 1))
        x = self.up2(x, x3, self.te_up2(t).reshape(n, -1, 1, 1))
        x = self.up3(x, x2, self.te_up3(t).reshape(n, -1, 1, 1))
        x = self.up4(x, x1, self.te_up4(t).reshape(n, -1, 1, 1))
        x = self.outc(x + self.te_out(t).reshape(n, -1, 1, 1))
        return x

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out))