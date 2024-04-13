import lmdb
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO


class LSUNDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.path = path

        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
            self.key_list = [key.decode("utf-8") for key, value in txn.cursor()]

        self.resolution = resolution
        self.transform = transform

        self.env.close()
        self.env = None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self.env = lmdb.open(
            self.path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with self.env.begin(write=False) as txn:
            #key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(self.key_list[index].encode("utf-8"))

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        self.env.close()
        self.env = None

        return img
