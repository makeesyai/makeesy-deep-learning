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
                 hidden_layer_size=128,
                 ):
        super(NeuralNet, self).__init__()
        self.deep_nn = nn.ParameterDict({
            'left': nn.Parameter(torch.rand(hidden_layer_size, input_size)),
            'right': nn.Parameter(torch.rand(hidden_layer_size, input_size))
        })
        self.classifier = nn.Parameter(torch.rand(output_size, hidden_layer_size))

    def forward(self, inputs, choice):
        # Linear: `y = xA^T + b`
        tensor = F.linear(inputs, self.deep_nn[choice])
        return F.linear(tensor, self.classifier)


model = NeuralNet(16, 2)
x_left = torch.rand(4, 16)
output_left = model(x_left, choice='left')
print(output_left)

x_right = torch.rand(4, 16)
output_right = model(x_right, choice='right')
print(output_right)

