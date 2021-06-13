# The torch.nn namespace provides all the building blocks you need to
# build your own neural network.
# Every module in PyTorch subclasses the nn.Module.

# How to correctly extend nn.Module?
import torch
from torch import nn


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()

    def forward(self, x):
        return self.flatten(x)


x = torch.rand(2, 2, 2)
print(x)
model = NeuralNet()
output = model(x)
print(output)
