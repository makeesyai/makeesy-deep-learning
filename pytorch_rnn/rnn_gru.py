# GRU: Gated Recurrent Unit
"""
Math:
r_t = \sigmoid(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr})
z_t = \sigmoid(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz})
n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn}))
h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}

1. No Memory cell
2. Only 2 gates, Update and Reset gates
3. Reset Gate: Input Gate and the Forget Gate of LSTM
4. Update Gate: Output Gate of LSTM

Points to remember
1. n_t: New Gate (pytorch), Current Memory Gate, Intermediate Hidden State, Candidate Hidden State etc.
2. Faster to train compared to LSTM with similar or better accuracy
"""

import torch
from torch import nn


class GRUNetworks(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUNetworks, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, bidirectional=False, batch_first=True)

    def forward(self, inp, hidden_state):
        return self.rnn(inp, hidden_state)


feature_size = 2
seq_length = 3
batch_size = 2

rnn_hidden_size = 3
rnn_num_layers = 1

model = GRUNetworks(feature_size, rnn_hidden_size, rnn_num_layers)

# (bs, seq-len/tokens, feature) if batch_first=True, otherwise (seq-len/tokens, bs, feature)
# For example input=['he is handsome', 'she is beautiful']
x = torch.randn(batch_size, seq_length, feature_size)

# print(list(model.named_parameters()))

hidden = torch.zeros(rnn_num_layers, batch_size, rnn_hidden_size)  # n_layers x bs x hidden_size
model_output, hidden = model(x, hidden)
# model_output: bs x seq x hidden_size, hidden: n_layers x bs x hidden_size

print(model_output)
print(hidden)
