# Loss functions:
# Loss functions are used to measure the error between the predicted values vs the provided target

# 1. Regression: used when the model is predicting a continuous value, e.g. house price
# 2. Classification: used when the model is predicting a discrete value, e.g. spam detection
# 3. Ranking: used when the model is predicting the relative distances between inputs, e.g. Information retrieval

# In this tutorial
# 1. Mean Absolute Error (L1 loss)
# loss(x, y) = |x - y| where x represents the actual value and y the predicted value.

# 2. Mean Squared Error (L2 loss)
# loss(x, y) = (x - y)^2 where x represents the actual value and y the predicted value.

import torch
import torch.nn as nn

pred_target = torch.randn(2, requires_grad=True)
target = torch.randn(2)
mae_loss = nn.L1Loss()
mse_loss = nn.MSELoss()
output_mae = mae_loss(pred_target, target)
output_mse = mse_loss(pred_target, target)
output_mae.backward()
output_mse.backward()

print('input: ', pred_target)
print('target: ', target)
print('output: ', output_mae)
print(abs(pred_target - target).sum()/target.shape[0])

print('output: ', output_mse)
print(pow(pred_target - target, 2).sum()/target.shape[0])

