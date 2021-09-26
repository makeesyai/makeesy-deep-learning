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
print(x_train, y_train)

# Simple Linear Regression: a + bx
a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)
lr = 0.1
# If we use nn.Module to create a model, it will model.parameters()
model = [Parameter(a.clone()), Parameter(b.clone())]

criterion = MSELoss()
optimizer = torch.optim.SGD(model, lr=lr)
# optimizer = torch.optim.Adam(model, lr=lr)

for epoch in range(500):
    # Remove the grad computed in the last step
    optimizer.zero_grad()
    print(f'model params')
    print(model[0])
    print(a)
    print(model[1])
    print(b)

    # Run a + bx
    y_predicted = model[0] + model[1] * x_train
    loss = criterion(y_predicted, y_train)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        # Let us compute manually
        y_predicted_manually = a + b * x_train
        error = (y_train - y_predicted_manually)
        loss_manual = (error ** 2).mean()
        # Computes gradients for both "a" and "b" parameters
        a_grad = -2 * error.mean()
        b_grad = -2 * (x_train * error).mean()

        # Updates parameters using gradients and the learning rate
        a = a - lr * a_grad
        b = b - lr * b_grad

    print(f'model predictions')
    print(y_predicted)
    print(y_predicted_manually)

    print(f'params grad')
    print(model[0].grad)
    print(a_grad)
    print(model[1].grad)
    print(b_grad)

    print(f'model loss')
    print(loss)
    print(loss_manual)

