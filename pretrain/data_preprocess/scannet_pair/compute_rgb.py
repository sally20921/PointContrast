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
from torch.nn import functional as F
from torchvision import transforms as transforms

from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform
from pl_bolts.models.self_supervised import SimSiam, SimCLR, BYOL, SwAV
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
from pl_bolts.models.self_supervised.simsiam.models import SiameseArm
from pl_bolts.models.self_supervised.resnets import resnet18, resnet50
from pl_bolts.optimizers.lars_scheduling import LARSWrapper

def _default_transforms():
        scannet_transforms = transforms.Compose([
            transforms.Resize((480,640)),
            transforms.ToTensor(),
        ])
        return scannet_transforms

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
        self.transform = transform
        self._imgs = sorted(glob.glob('/home/data/scannet/scans/'+'*/color/*.jpg'))
        #print(self._imgs)

    def __len__(self):
        #print(len(self._imgs))
        return len(self._imgs)

    def __getitem__(self, idx):
        img = Image.open(self._imgs[idx])
        #img = np.array(img)

        if self.transform is not None:
            x_i = self.transform(img)
            x_j = self.transform(img)
        #print(x_i)
        #print(x_j)
        return (x_i, x_j, idx), idx


class ScannetDataModule(LightningDataModule):
    name = 'scannet'
    dataset_cls = ScannetDataset
    dim = (3, 480, 640)
    def __init__(self,
            data_dir: Optional[str] = None,
            val_split: float = 0.2,
            test_split: float = 0.1,
            num_workers: int = 4,
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
                transforms.Resize((480,640)),
                transforms.ToTensor(),
        ])
        return scannet_transforms

def cosine_similarity(a, b):
    b = b.detach()
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    sim = -1 * (a * b).sum(-1).mean()
    return sim

def main():
    #print(sorted(glob.glob('home/data' + '/*/')))
    dm = ScannetDataModule('/home/data/scannet')
    #dm = CIFAR10DataModule(data_dir='/home/data', batch_size=32, num_workers=0, val_split=5000)
    
    dm.train_transforms = SimCLRTrainDataTransform(
            input_height= 640,
            gaussian_blur=True,
            jitter_strength=1.0,
            normalize=imagenet_normalization,
    )
    dm.val_transforms = SimCLREvalDataTransform(
            input_height=640,
            gaussian_blur=True,
            jitter_strength=1.0,
            normalize=imagenet_normalization,
    )

    # init 
    #backbone = resnet50
    #encoder = backbone(first_conv=True, maxpool1=True, return_all_feature_maps=False)
    #online_network = SiameseArm(
    #        encoder, input_dim=2048, hidden_size=2048, output_dim=128
    #)

    
    #train_dataloader = dm.train_dataloader()
    #val_dataloader = dm.val_dataloader()

    #optimizer = torch.optim.SGD(online_network.params(), lr=0.001, momentum=0.9, weight_decay=0.0001)

    #for idx, (img1, img2) in enumerate(train_dataloader):
    #    _, z1, h1 = online_network(img1)
    #    _, z2, h2 = online_network(img2)
    #    loss = cosine_similarity(h1, z2) / 2+cosine_similarity(h2, z1) / 2
        #optimizer.step()

        #print(idx)
        #print(img1)
        #print(img2)

    model = SimSiam(gpus=8, num_samples=2474251, batch_size=32, dataset='scannet')  
    online_evaluator = SSLOnlineEvaluator(
            drop_p=0.0,
            hidden_dim=None,
            z_dim=2058,
            num_classes=1000,
            dataset='scannet'
    )

    trainer = pl.Trainer()
    trainer.fit(model, datamodule=dm)

if __name__=="__main__":
    main()

