from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader


def get_datasets(train_dir, test_dir, train_transform, test_transform):
    train_dataset = ImageFolder(train_dir, train_transform)
    test_dataset = ImageFolder(test_dir, test_transform)

    return train_dataset, test_dataset

def get_loaders(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

    return train_loader, test_loader
