import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from typing import Optional, Union, List


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 transform_mean: Union[float, List[float]],
                 transform_std: Union[float, List[float]],
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 **kwargs):
        super().__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.size = 32
        self.n_channels = 1

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(transform_mean, transform_std)
        ])

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = MNIST("data/", download=True, train=True, transform=self.transform)
        self.val_dataset = MNIST("data/", download=True, train=False, transform=self.transform)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.train_batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.val_batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)
