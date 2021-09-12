# https://gombru.github.io/2019/04/03/ranking_loss/

# Ranking Loss: The objective of Ranking Losses is to predict relative distances between inputs.
# 1. Pair Ranking Loss
# 2. Triplet Ranking Loss

# Pair Ranking Loss Formulation
# L(r0, r1, y) = max(0, margin - y*||r0 - r1||)
# Where r0, and r1 are the input features and
# y = 1, and -1 for positive and negative examples respectively.

# Triplet Ranking Loss Formulation
# L(a, p, n) = max(|| a - p || - || a - n || + margin, 0)
# where a (anchor), p (positive examples), and n (negative examples)
import torch
from torch import nn

loss = nn.MarginRankingLoss()
loss_triplet = nn.TripletMarginLoss()
input1 = torch.randn(3, 512, requires_grad=True)  # Anchor
input2 = torch.randn(3, 512, requires_grad=True)  # Positive
input3 = torch.randn(3, 512, requires_grad=True)  # Negative
target = torch.randn(3, 1).sign()
output = loss(input1, input2, target)
output.backward()

output_triplet = loss_triplet(input1, input2, input3)
output_triplet.backward()

print(input1)
print(input2)
print(input3)
print(target)
print(output)
print(output_triplet)
