import torch

x = torch.tensor([2., 3.], requires_grad=True)
y = torch.tensor([5., 5.], requires_grad=True)

cost_fn = (x - y) ** 2
print(cost_fn)
# J(x,y) = (x-y) ** 2
# dJ/dx = 2(x-y)
# dJ/dy = -2(x-y)

# cost_fn.sum().backward()

external_grad = torch.tensor([1., 1.])
cost_fn.backward(gradient=external_grad)
print(x.grad)
print(y.grad)

# check if collected gradients are correct
print(2*(x-1) == x.grad)
print(2*(y-1) == y.grad)
