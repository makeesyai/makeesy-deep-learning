from numpy import matmul, round_
from scipy.special import softmax

Q = [
    [0, 1, 1],  # Query for input 1
    [4, 6, 0],  # Query for input 2
    [2, 3, 1],  # Query for input 3
]
K_transpose = [
    [1, 4, 3],
    [1, 2, 1],
    [1, 2, 2],
]
attn_scores = [
    [2, 4, 3],
    [10, 28, 18],
    [6, 16, 11],
]
softmax_attn_score = [
    [0.1, 0.7, 0.2],  # Scores for query 1 [k1, k2, k3]
    [0.0, 1.0, 0.0],  # Scores for query 2 [k1, k2, k3]
    [0.0, 1.0, 0.0],  # Scores for query 3 [k1, k2, k3]
]

print(matmul(Q, K_transpose))
print(round_(softmax(attn_scores, axis=-1), decimals=1))
