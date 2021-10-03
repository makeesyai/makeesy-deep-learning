# Elman RNN
"""
Math: h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})
"""


import torch
from torch import nn


class ElmanRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(ElmanRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, bidirectional=False, batch_first=True)

    def forward(self, inp, hidden):
        return self.rnn(inp, hidden)


input_size = 2
seq_length = 5
batch_size = 4

hidden_size = 3
num_layers = 1

model = ElmanRNN(input_size, hidden_size, num_layers)

# (bs, seq-len/tokens, feature) if batch_first=True, otherwise (seq-len/tokens, bs, feature)
# For example input=['he is handsome', 'she is beautiful']
x = torch.randn(batch_size, seq_length, input_size)

# print(list(model.named_parameters()))
hidden = torch.zeros(num_layers, batch_size, hidden_size)  # n_layers x bs x hidden_size
model_output, hidden = model(x, hidden)
# model_output: bs x seq x hidden_size, hidden: n_layers x bs x hidden_size

print(model_output)
print(hidden)
