from typing import Union, List

from datamodules.lsun_datamodule import LSUNDataModule
from datamodules.mnist_datamodule import MNISTDataModule
from datamodules.cifar10_datamodule import CIFAR10DataModule


def datamodule_selector(dataset: str,
                        dataset_path: str,
                        train_batch_size: int,
                        val_batch_size: int,
                        shuffle: bool,
                        num_workers: int,
                        transform_mean: Union[float, List[float]],
                        transform_std: Union[float, List[float]]):
    if dataset == "mnist":
        return MNISTDataModule(root=dataset_path,
                               train_batch_size=train_batch_size,
                               val_batch_size=val_batch_size,
                               shuffle=shuffle,
                               num_workers=num_workers,
                               transform_mean=transform_mean,
                               transform_std=transform_std)
    elif dataset == "lsun_churches":
        return LSUNDataModule(root=dataset_path,
                              train_batch_size=train_batch_size,
                              val_batch_size=val_batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              transform_mean=transform_mean,
                              transform_std=transform_std)
    elif dataset == "cifar10":
        return CIFAR10DataModule(root=dataset_path,
                              train_batch_size=train_batch_size,
                              val_batch_size=val_batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              transform_mean=transform_mean,
                              transform_std=transform_std)
    else:
        raise Exception("Invalid dataset.")
    