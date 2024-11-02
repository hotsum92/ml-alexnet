from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import torchinfo
from matplotlib import pyplot as plt
import torchvision.models as models
from alexnet import AlexNet
from train import train_eval
import sys

#image_size = 32
image_size = int(sys.argv[1])
#batch_size = 100
batch_size = int(sys.argv[2])
#num_epochs = 100
num_epochs = int(sys.argv[3])

num_classes = 10

model=AlexNet(num_classes)

torchinfo.summary(model, (1, 3, image_size, image_size))

loss_func = F.cross_entropy
optimizer = optim.Adam(model.parameters())

normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2470, 0.2435, 0.2616],
)

train_transform = transforms.Compose([
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
    batch_size,
    shuffle=True,
)

print("train_loader", len(train_loader))

test_loader = DataLoader(
    test_dataset,
    batch_size,
    shuffle=False,
)

print("test_loader", len(test_loader))

train_eval(f"i{image_size}b{batch_size}", model, num_epochs, train_loader, test_loader, loss_func, optimizer)
