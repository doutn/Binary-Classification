from dataloader import get_datasets, get_loaders
from trainer import Trainer
from utils import get_transforms
from model import VGG_net

import torch
import torch.nn as nn
import argparse
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def parse():
    parser = argparse.ArgumentParser(prog="binary_classification", description="cat_and_dog")
    parser.add_argument("--train_dir", type=str, action="store", default="./data/training_set/")
    parser.add_argument("--test_dir", type=str, action="store", default="./data/test_set/")
    parser.add_argument("--batch_size", type=int, action="store", default=64)
    parser.add_argument("--learning_rate", type=int, action="store", default=1e-3)
    parser.add_argument("--num_epochs", type=int, action="store", default=20)
    parser.add_argument("--device", type=str, action="store", default="mpu")
    parser.add_argument("--output_dir", type=str, action="store", default="./output/")

    args = parser.parse_args()
    return args


def process(args):
    train_transform, test_transform = get_transforms()
    train_datasets, test_datasets = get_datasets(args.train_dir, args.test_dir, train_transform, test_transform)
    train_loader, test_loader = get_loaders(train_datasets, test_datasets, args.batch_size)

    model = VGG_net(3, 2)

    if args.device == "mpu":
        device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
    
    else:
        device = torch.device('cuda:0' if torch.backends.mps.is_available() else 'cpu')

    loss_fn = nn.CrossEntropyLoss()
    n_steps = len(train_datasets) // args.batch_size
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=(n_steps * args.num_epochs), eta_min=0)

    trainer = Trainer(train_loader, test_loader, model, loss_fn, optimizer, scheduler, device, args.output_dir)
    trainer.train(args.num_epochs)


if __name__ == "__main__":
    args = parse()
    process(args)
