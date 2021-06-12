# forward hook (executing after the forward pass),
# register_forward_hook(hook)
import torch
from torch import nn
from torch.nn.utils import prune


class ModelNet(nn.Module):
    def __init__(self, input_size,
                 num_hidden_layers=1,
                 hidden_layer_size=512,
                 num_labels=2,
                 ):
        super(ModelNet, self).__init__()
        self.model = nn.Sequential()
        for i in range(num_hidden_layers):
            self.model.add_module(f'ff_{i}', nn.Linear(input_size, hidden_layer_size))
            self.model.add_module(f'relu{i}', nn.ReLU())
            input_size = hidden_layer_size
        self.model.add_module('classification', nn.Linear(hidden_layer_size, num_labels))

    def forward(self, x):
        return self.model(x)


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        # print(module)
        # print(module_in[0].shape)
        # print(module_out.shape)
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


x = torch.rand(16, 128)
print(x)
input_size = x.shape[1]
model = ModelNet(input_size=input_size)

save_output = SaveOutput()
hook_handles = []
for layer in model.modules():
    if isinstance(layer, torch.nn.modules.Linear) or \
            isinstance(layer, torch.nn.modules.ReLU):
        handle = layer.register_forward_hook(save_output)
        hook_handles.append(handle)

out = model(x)
print(out.size())
print(len(save_output.outputs))
for i in range(len(save_output.outputs)):
    print(save_output.outputs[i].shape)


# Understanding Pruning
module = model.model
print(list(module.named_parameters()))
print(list(module.named_buffers()))
prune.random_unstructured(module[0], name="weight", amount=0.3)
prune.random_unstructured(module[0], name="bias", amount=0.3)
print(list(module.named_parameters()))
print(list(module.named_buffers()))
print(module[0].weight)
print(module[0]._forward_pre_hooks)
