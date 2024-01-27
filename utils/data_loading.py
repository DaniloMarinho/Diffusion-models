import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, ImageFolder
from torchvision import transforms


def load_dataset(dataset=None, dataset_path=None):
    assert dataset is not None or dataset_path is not None, "No valid dataset selected"
    train_dataset = None

    if dataset == "MNIST":
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        train_dataset = MNIST("data/", download=True, train=True, transform=transform)
    elif dataset is None:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        train_dataset = ImageFolder(dataset_path, transform=transform)

    return train_dataset
