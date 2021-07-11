# Containers
# Sequential : A sequential container.
# ModuleList: Holds submodules in a list.
# ModuleDict: Holds submodules in a dictionary.
# ParameterList: Holds parameters in a list.
# ParameterDict: Holds parameters in a dictionary.
import torch
from torch import nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size, num_hidden_layers=10, hidden_layer_size=128):
        super(NeuralNet, self).__init__()
        self.deep_nn = nn.Sequential()
        for i in range(num_hidden_layers):
            self.deep_nn.add_module(f'ff{i}', nn.Linear(input_size, hidden_layer_size))
            self.deep_nn.add_module(f'activation{i}', nn.ReLU())
            input_size = hidden_layer_size
        self.deep_nn.add_module(f'classifier', nn.Linear(hidden_layer_size, output_size))

    def forward(self, inputs):
        tensor = self.deep_nn(inputs)
        return tensor


x = torch.rand(4, 16)
model = NeuralNet(16, 2)
output = model(x)
print(output)
