from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
from lightning import LightningDataModule
import os

class MyDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        ds = torchvision.datasets.CIFAR10
        path = os.path.join("./data", "CIFAR10")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                (0.2023, 0.1994, 0.2010)),
        ])
        self.train_set = ds(path, train=True, download=True, transform=transform_train)
        self.test_set = ds(path, train=False, download=True, transform=transform_test)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=7, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=7, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=7, persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=7, persistent_workers=True)