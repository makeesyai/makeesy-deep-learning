# Loss functions
# 1. Regression
# 2. Classification
# 3. Ranking

import torch
import torch.nn as nn

pred_target = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)

mae_loss = nn.L1Loss()
output = mae_loss(pred_target, target)
output.backward()

print('input: ', pred_target)
print('target: ', target)
print('output: ', output)
