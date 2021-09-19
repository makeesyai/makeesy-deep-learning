# SGD: Stochastic Gradient Descent
# Useful links
# https://ruder.io/optimizing-gradient-descent/
# https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e

# STEPS
# 1. Compute Loss
# 2. Compute Partial Derivatives
# 3. Update the Parameters
# 4. Repeat 1-3 until convergence


import torch
from torch.nn import MSELoss, Parameter

# The data function is: y = x + 10
x_train = torch.tensor([1, 2, 3, 4])
y_train = torch.tensor([11, 12, 13, 14], dtype=torch.float)

# Simple Linear Regression: a + bx
a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)

# If we use nn.Module to create a model, it will model.parameters()
model = [Parameter(a), Parameter(b)]

criterion = MSELoss()
# optimizer = torch.optim.SGD(model, lr=0.1)
optimizer = torch.optim.Adam(model, lr=0.1)

for epoch in range(500):
    # Remove the grad computed in the last step
    optimizer.zero_grad()
    # Run a + bx
    y_predicted = model[0] + model[1] * x_train
    print(x_train, y_train, y_predicted)
    loss = criterion(y_predicted, y_train)
    loss.backward()
    print(loss)
    optimizer.step()
