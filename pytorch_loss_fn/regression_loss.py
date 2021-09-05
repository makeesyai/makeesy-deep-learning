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
from torch import nn

predicted_target = torch.randn(2, requires_grad=True)
target = torch.randn(2)
mae_loss = abs(target - predicted_target).sum() / target.shape[0]
mse_loss = pow(target - predicted_target, 2).sum()/target.shape[0]

l1_loss = nn.L1Loss()
l2_loss = nn.MSELoss()

mae_output = l1_loss(predicted_target, target)
mse_output = l2_loss(predicted_target, target)

mae_output.backward()
mse_output.backward()


print(mae_loss)
print(mae_output)

print(mse_loss)
print(mse_output)



