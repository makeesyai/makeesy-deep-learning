# Containers
# Sequential : A sequential container.
# ModuleList: Holds submodules in a list.
# ModuleDict: Holds submodules in a dictionary.
# ParameterList: Holds parameters in a list.
# ParameterDict: Holds parameters in a dictionary.
import torch
from torch import nn
from torch.nn import functional as F


class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size,
                 num_hidden_layers=10,
                 hidden_layer_size=128,
                 ):
        super(NeuralNet, self).__init__()
        self.activation = nn.ReLU()
        self.deep_nn = nn.ParameterList()
        for i in range(num_hidden_layers):
            self.deep_nn.append(nn.Parameter(torch.rand(hidden_layer_size, input_size)))
            input_size = hidden_layer_size
        self.deep_nn.append(nn.Parameter(torch.rand(output_size, hidden_layer_size)))

    def forward(self, inputs):
        hidden_states = []
        for idx, layer in enumerate(self.deep_nn):
            # Linear: `y = xA^T + b`
            tensor = F.linear(inputs, layer)
            if idx != len(self.deep_nn) - 1:
                tensor = self.activation(tensor)
            hidden_states.append(tensor)
            inputs = tensor
        return hidden_states[-1], hidden_states


x = torch.rand(4, 16)
model = NeuralNet(16, 2)
output, states = model(x)
print(output)
