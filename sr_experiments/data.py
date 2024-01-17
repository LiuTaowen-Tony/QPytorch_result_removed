# loading data
import os
import torch
import torchvision
import torchvision.transforms as transforms


def get_loaders(batch_size):
    ds = torchvision.datasets.CIFAR10
    path = os.path.join("./data", "CIFAR10")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = ds(path, train=True, download=True, transform=transform_train)
    test_set = ds(path, train=False, download=True, transform=transform_test)
    return {
            'train': torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            ),
            'test': torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True
            )
    }