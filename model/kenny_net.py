import torch
from torch import nn
from torch.nn import functional as F


class KennyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 5, padding='same')
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        return x


net = KennyNet()

inp = torch.randn((10, 3, 500, 500))

out = net(inp)
print(out.shape)