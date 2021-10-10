# RNN Limitations
# 1. Long Range Dependencies
# 2. Vanishing Gradient

# LSTM
# Solves the above limitation using Gates
"""
i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
h_t = o_t \odot \tanh(c_t) \\
"""

import torch
from torch import nn


class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMNetwork, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=False, batch_first=True)

    def forward(self, inp, hidden_state):
        return self.rnn(inp, hidden_state)


feature_size = 2
seq_length = 3
batch_size = 2

rnn_hidden_size = 3
rnn_num_layers = 1

model = LSTMNetwork(feature_size, rnn_hidden_size, rnn_num_layers)

# (bs, seq-len/tokens, feature) if batch_first=True, otherwise (seq-len/tokens, bs, feature)
# For example input=['he is handsome', 'she is beautiful']
x = torch.randn(batch_size, seq_length, feature_size)

# xt = x.transpose(1, 2)
# print(torch.matmul(model.rnn.weight_ih_l0, xt))

hidden = torch.zeros(rnn_num_layers, batch_size, rnn_hidden_size)  # n_layers x bs x hidden_size
model_output, hidden = model(x, (hidden, hidden))
# model_output: bs x seq x hidden_size, ht, ct: n_layers x bs x hidden_size

print(model_output)
print(hidden[0].shape, hidden[1].shape)  # ht, ct
