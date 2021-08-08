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
            'class_1': nn.Parameter(torch.rand(hidden_layer_size, input_size)),
            'class_2': nn.Parameter(torch.rand(hidden_layer_size, input_size))
        })
        self.classify = nn.Parameter(torch.rand(output_size, hidden_layer_size))

    def forward(self, inputs, choice):
        return F.linear(F.linear(inputs, self.deep_nn[choice]), self.classify)


model = NeuralNet(16, 2)

x_class1 = torch.rand(4, 16)
output_class1 = model(x_class1, 'class_1')

x_class2 = torch.rand(4, 16)
output_class2 = model(x_class2, 'class_2')

print(output_class1)
print(output_class2)
