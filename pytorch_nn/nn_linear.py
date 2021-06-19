# nn.Linear: Applies a linear transformation to the incoming data: y = xA^T + b

# torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
# * in_features – size of each input sample
# * out_features – size of each output sample
# * bias – If set to False, the layer will not learn an additive bias. Default: True

# Input: (N, *, H_in), where H_in=in_features
# Output: (N, *, H_out) where H_out = =out_features.
import torch
from torch import nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.ff = nn.Linear(input_size, output_size)
        print(self.ff.weight.shape)
        print(self.ff.bias.shape)

    def forward(self, inputs):
        pass


model = NeuralNet(16, 2)
# x = torch.rand(4, 16)
# model_output = model(x)

