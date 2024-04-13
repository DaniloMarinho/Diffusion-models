import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from typing import Optional

from datasets.lsun_dataset import LSUNDataset


class LSUNDataModule(pl.LightningDataModule):
    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
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
        
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.train_transform = transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = LSUNDataset(path=self.root, transform=self.train_transform)
        self.val_dataset = LSUNDataset(path=self.root, transform=self.val_transform)
        
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
