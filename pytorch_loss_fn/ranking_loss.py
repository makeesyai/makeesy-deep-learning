# https://gombru.github.io/2019/04/03/ranking_loss/

# Ranking Loss: The objective of Ranking Losses is to predict relative distances between inputs.
# 1. Pair Ranking Loss
# 2. Triplet Ranking Loss

# Pair Ranking Loss Formulation (MarginRankingLoss)
# L(r0, r1, y) = max(0, margin - y*||r0 - r1||)
# Where r0, and r1 are the input features and
# y = 1, and -1 for positive and negative examples respectively.

# Triplet Ranking Loss Formulation (TripletMarginLoss)
# L(a, p, n) = max(|| a - p || - || a - n || + margin, 0)
# where a (anchor), p (positive examples), and n (negative examples)
import torch
from torch import nn

input1 = torch.rand(3, 512, requires_grad=True)
input2 = torch.rand(3, 512, requires_grad=True)
input3 = torch.rand(3, 512, requires_grad=True)

loss_fn_pair = nn.MarginRankingLoss()
loss_fn_triplet = nn.TripletMarginLoss()
target = torch.randn(3,1).sign()
output_pair = loss_fn_pair(input1, input2, target)
output_pair.backward()
output_triplet = loss_fn_triplet(input1, input2, input3)
output_triplet.backward()

print(input1)
print(input2)
print(input3)
print(output_pair)
print(output_triplet)
