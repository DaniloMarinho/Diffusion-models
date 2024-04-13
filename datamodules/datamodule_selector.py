from datamodules.lsun_datamodule import LSUNDataModule
from datamodules.mnist_datamodule import MNISTDataModule


def datamodule_selector(dataset: str,
                        dataset_path: str,
                        train_batch_size: int,
                        val_batch_size: int,
                        shuffle: bool,
                        num_workers: int):
    if dataset == "mnist":
        return MNISTDataModule(root=dataset_path,
                               train_batch_size=train_batch_size,
                               val_batch_size=val_batch_size,
                               shuffle=shuffle,
                               num_workers=num_workers)
    elif dataset == "lsun_churches":
        return LSUNDataModule(root=dataset_path,
                              train_batch_size=train_batch_size,
                              val_batch_size=val_batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers)
    else:
        raise Exception("Invalid dataset.")
    