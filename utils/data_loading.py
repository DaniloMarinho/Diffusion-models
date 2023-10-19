import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms


def load_dataset(dataset="MNIST"):
    train_dataset = None

    if dataset == "MNIST":
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        train_dataset = MNIST("data/", download=True, train=True, transform=transform)

    return train_dataset
