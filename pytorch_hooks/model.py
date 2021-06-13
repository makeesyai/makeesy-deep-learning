from torch import nn


class ModelNet(nn.Module):
    def __init__(self, input_size,
                 num_hidden_layers=1,
                 hidden_layer_size=512,
                 num_labels=2,
                 ):
        super(ModelNet, self).__init__()
        self.model = nn.Sequential()
        for i in range(num_hidden_layers):
            self.model.add_module(f'ff_{i}', nn.Linear(input_size, hidden_layer_size))
            self.model.add_module(f'relu{i}', nn.ReLU())
            input_size = hidden_layer_size
        self.model.add_module('classification', nn.Linear(hidden_layer_size, num_labels))

    def forward(self, x):
        return self.model(x)
