import torch
from torch import optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import torchsummary
import torchinfo
from matplotlib import pyplot as plt
import torchvision.models as models
from alexnet import AlexNet
from train import train_eval

normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2470, 0.2435, 0.2616],
)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
    ),
    transforms.ToTensor(),
    normalize,
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform,
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
)

num_classes = 10
num_epochs = 100

model=AlexNet(num_classes)

loss_func = F.cross_entropy
optimizer = optim.Adam(model.parameters())

torchinfo.summary(model, (1, 3, 32, 32))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_eval(model, num_epochs, train_loader, test_loader, loss_func, optimizer)
