# backward hook (executing after the backward pass).
# register_backward_hook(hook)
from typing import Union

import torch
from torch import nn, Tensor
from torch.nn import Module

from pytorch_hooks.forward_hook import ModelNet


class LinearLayerUpdateGradient:
    def __init__(self, amount):
        self.amount = amount

    def __call__(self, module: Module, grad_input: Union[tuple[Tensor, ...], Tensor],
                 grad_output: Union[tuple[Tensor, ...], Tensor]):
        updated_grad = []
        for idx in range(len(grad_input)):
            mask = torch.FloatTensor(grad_input[idx].size()).uniform_() > self.amount
            updated_grad.append(grad_input[0].masked_fill_(mask, value=0.0))
        return tuple(updated_grad)


x = torch.rand(2, 4)
print(x)
inp_dim = x.shape[1]
model = ModelNet(input_size=inp_dim)
backward_hook = LinearLayerUpdateGradient(amount=0.2)
hooks = []
for sub_module in model.modules():
    if isinstance(sub_module, nn.modules.Linear):
        print(sub_module)
        # Models with dicts as output must use register_full_backward_hook
        if hasattr(sub_module, "register_full_backward_hook"):
            hook = sub_module.register_full_backward_hook(backward_hook)
        else:
            hook = sub_module.register_backward_hook(backward_hook)
        hooks.append(hook)


output = model(x)
output.mean().backward()
print(output)
