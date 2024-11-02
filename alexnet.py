import torch

class AlexNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = torch.nn.Sequential(
            # output size = (input_size - kernel_size + 2*padding)/stride + 1
            # size 32 -> (32-3+2*1)/1 + 1 = 32
            # Conv2d [1, 64, 32, 32]
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            # ouput size = (input_size - kernel_size)/stride + 1
            # size 32 -> (32-2)/2 + 1 = 16
            # MaxPool2d [1, 64, 16, 16]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(64),
            # size 16 -> (16-5+2*2)/1 + 1 = 16
            # Conv2d [1, 192, 16, 16]
            torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            # size 16 -> (16-2)/2 + 1 = 8
            # MaxPool2d [1, 192, 8, 8]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(192),
            # size 8 -> (8-3+2*1)/1 + 1 = 8
            # Conv2d [1, 384, 8, 8]
            torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            # size 8 -> (8-3+2*1)/1 + 1 = 8
            # Conv2d [1, 256, 8, 8]
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            # size 8 -> (8-3+2*1)/1 + 1 = 8
            # Conv2d [1, 256, 8, 8]
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            # size 8 -> (8-2)/2 + 1 = 4
            # MaxPool2d [1, 256, 4, 4]
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(256),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(256 * 4 * 4, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
