# Dropout:
# 1. Proven to be an effective technique for regularization.
# 2. Usage samples from a Bernoulli distribution.
# 3. In a batch, each sample will be zeroed out independently.
# 4. The outputs are scaled by a factor of 1/1−p.
# 5. During evaluation the module simply computes an identity function.


# Bernoulli sampling: Basics
import torch
from torch import nn
from torch.distributions import Bernoulli

p = 0.3
bernoulli_sampler = Bernoulli(probs=p)  # 30% chance 1; 70% chance 0
samples = bernoulli_sampler.sample((20,))
# Compute % of 1
print(samples.count_nonzero()/len(samples))

# Create dropout layer
dropout_model = nn.Dropout(p=p)  # Default: mode train
tensor = torch.randn(4, 2, 10)
print(tensor)
print(tensor * (1/(1-p)))  # Scaling tensor
output = dropout_model(tensor)
print(output)

# Eval mode
dropout_model = dropout_model.eval()
output = dropout_model(tensor)
print(output)

# Train mode
dropout_model = dropout_model.train()
output = dropout_model(tensor)
print(output)
