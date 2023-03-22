from torchvision import transforms
from model import VGG_net

import torch
import cv2
import torch.nn as nn


def get_transforms():
    train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(80, 80))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(80, 80))
    ])

    return train_transform, test_transform


def predict(image_dir, model_pth_dir):
    model = VGG_net(3, 2)
    model.load_state_dict(torch.load(model_pth_dir))

    image = cv2.imread(image_dir, cv2.IMREAD_COLOR)

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(80, 80))    
    ])

    image = transform(image).unsqueeze(0)

    output = model(image)
    sigmoid = nn.Sigmoid()
    output = sigmoid(output)

    if output[0][0] < output[0][1]:
        print("강아지")

    else:
        print("고양이")
    