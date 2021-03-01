import os, glob
import numpy as np
from PIL import Image
from typing import Any, Callable, Optional

from torch.utils.data import Dataset
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import pytorch_lightning as pl

from torchvision import transforms as transforms

from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform
from pl_bolts.models.self_supervised import SimSiam
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization

class ScannetDataset(Dataset):

    IMAGE_PATH = os.path.join('/home/data', 'scannet')

    def __init__(
            self,
            data_dir: str,
            img_size: tuple = (1296, 968),
            transform=None,
    ):
        """
        data_dir
        img_size = (1296, 968)

        """
        self.img_size = img_size

        self._imgs = sorted(glob.glob('/home/data/scannet/scans/'+'*/color/*.jpg'))
        #print(self._imgs)

    def __len__(self):
        #print(len(self._imgs))
        return len(self._imgs)

    def __getitem__(self, idx):
        img = Image.open(self._imgs[idx])
        img = np.array(img)

        if self.transform:
            img = self.transform(img)

        return img


class ScannetDataModule(LightningDataModule):
    name = 'scannet'
    def __init__(self,
            data_dir: Optional[str] = None,
            val_split: float = 0.2,
            test_split: float = 0.1,
            num_workers: int = 16,
            batch_size: int= 32,
            seed: int = 42,
            shuffle: bool = True,
            pin_memory: bool = True,
            drop_last: bool = False,
            *args: Any,
            **kwargs: Any,
    ) -> None:

        super().__init__(*args, **kwargs)

        self.data_dir = os.path.join('/home/data', 'scannet')
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        scannet_dataset = ScannetDataset(data_dir='/home/data/scannet', transform=self._default_transforms())
        #print(val_split)
        print(len(scannet_dataset))
        val_len = round(float(val_split) * len(scannet_dataset))
        test_len = round(float(test_split) * len(scannet_dataset))
        train_len = len(scannet_dataset) - val_len - test_len

        self.trainset, self.valset, self.testset = random_split(
            scannet_dataset, lengths=[train_len, val_len, test_len], generator=torch.Generator().manual_seed(self.seed))

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(
                self.trainset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(
                self.valset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                pin_memory=self.pin_memory
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(
                self.testset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                pin_memory=self.pin_memory,
        ) 
        return loader

    def _default_transforms(self) -> Callable:
        scannet_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])


def main():
    #print(sorted(glob.glob('home/data' + '/*/')))
    dm = ScannetDataModule('/home/data/scannet')
    dm.train_transforms = SimCLRTrainDataTransform(
            input_height= 968,
            gaussian_blur=True,
            jitter_strength=1.0,
            normalize=imagenet_normalization,
    )
    dm.val_transforms = SimCLREvalDataTransform(
            input_height=968,
            gaussian_blur=True,
            jitter_strength=1.0,
            normalize=imagenet_normalization,
    )


    model = SimSiam(gpus=8, num_samples=2474251, batch_size=32, dataset='scannet')

    pl.Trainer().fit(model, datamodule=dm)

if __name__=="__main__":
    main()

