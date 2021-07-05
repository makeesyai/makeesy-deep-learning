# Containers
# Sequential : A sequential container.
# ModuleList: Holds submodules in a list.
# ModuleDict: Holds submodules in a dictionary.
# ParameterList: Holds parameters in a list.
# ParameterDict: Holds parameters in a dictionary.
import torch
from torch import nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size,
                 num_hidden_layers=10,
                 hidden_layer_size=128,
                 choice='rrelu'):
        super(NeuralNet, self).__init__()

        self.activations = nn.ModuleDict({
            'rrelu': nn.ReLU(),
            'prelu': nn.PReLU(),
        })

        self.deep_nn = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.deep_nn.add_module(f'ff{i}', nn.Linear(input_size, hidden_layer_size))
            self.deep_nn.add_module(f'activation', self.activations[choice])
            input_size = hidden_layer_size
        self.deep_nn.add_module(f'classifier', nn.Linear(hidden_layer_size, output_size))

    def forward(self, inputs):
        hidden_states = []
        for layer in self.deep_nn:
            tensor = layer(inputs)
            hidden_states.append(tensor)
            inputs = tensor
        return hidden_states[-1], hidden_states


x = torch.rand(4, 16)
model = NeuralNet(16, 2, choice='prelu')
output, hidden_states = model(x)
print(output)
