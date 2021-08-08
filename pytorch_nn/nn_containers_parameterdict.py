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
        self.deep_nn = nn.ParameterDict()
        self.deep_nn['weights_hidden_layer'] = nn.Parameter(torch.rand(hidden_layer_size, input_size))
        self.deep_nn['weights_classifier_layer'] = nn.Parameter(torch.rand(output_size, hidden_layer_size))

    def forward(self, inputs, choice):
        return F.linear(inputs, self.deep_nn[choice])


x = torch.rand(4, 16)
model = NeuralNet(16, 2)
output = model(x, 'weights_hidden_layer')
output_class = model(output, 'weights_classifier_layer')
print(output_class)
