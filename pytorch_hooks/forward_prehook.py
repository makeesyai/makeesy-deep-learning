# forward prehook (executing before the forward pass)
# register_forward_pre_hook(hook)
import torch
from torch import nn
from torch.nn.utils import prune

from pytorch_hooks.model import ModelNet


class PruneLinearLayer:
    def __init__(self, param_names, amount):
        self.param_names = param_names
        self.amount = amount

    def __call__(self, module, module_in):
        print('Module inputs before')
        print(module_in)
        for p_name, a in zip(self.param_names, self.amount):
            prune.random_unstructured(module, p_name, amount=a)
        print('Module buffers')
        print(list(module.named_buffers()))
        print('Module params')
        print(list(module.named_parameters()))
        print('Module inputs')
        print(module_in)


if __name__ == '__main__':
    x = torch.rand(2, 4)
    print(x)
    inp_dim = x.shape[1]
    save_model = PruneLinearLayer(param_names=['weight', 'bias'], amount=[0.8, 0.2])
    model = ModelNet(input_size=inp_dim, num_hidden_layers=1)
    hooks = []
    for idx, layer in enumerate(model.modules()):
        if isinstance(layer, nn.modules.Linear):
            hook = layer.register_forward_pre_hook(save_model)
            hooks.append(hook)

    output = model(x)
    print(output)
