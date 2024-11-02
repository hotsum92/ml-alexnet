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
image_size = sys.args[0]
#batch_size = 128
batch_size = sys.args[1]
#num_epochs = 100
num_epochs = sys.args[2]

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
    batch_size=batch_size
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)

train_eval(model, num_epochs, train_loader, test_loader, loss_func, optimizer)
