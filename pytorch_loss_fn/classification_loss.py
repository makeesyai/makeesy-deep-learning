# Classification Loss functions-
# 1. Negative Log-Likelihood Loss Function
# 2. Cross Entropy Function

# Conceptually negative log likelihood and cross entropy are the same.

# 1. Used with a model having softmax as the lass layer for Multi-class classification problems.
# 2. As the output of softmax layer is between 0, and 1, and the log will be negative.
# So the negative sign is mainly to make the end loss positive (which we minimize eventually in training process).
# 3. Not only make the model to predict correctly but also with high confidence.

import torch
from torch import nn

predictions = torch.randn(3, 5, requires_grad=True)
print(predictions)
target = torch.tensor([0, 1, 4])  # 0<= values < C
print(target)

loss_fn_nll = nn.NLLLoss()
loss_fn_ce = nn.CrossEntropyLoss()

log_softmax = nn.LogSoftmax(dim=-1)
loss_nll = loss_fn_nll(log_softmax(predictions), target)
loss_ce = loss_fn_ce(predictions, target)
loss_nll.backward()
loss_ce.backward()

print(loss_nll)
print(loss_ce)
