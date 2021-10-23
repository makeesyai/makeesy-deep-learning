from numpy import matmul

x = [
    [1, 0, 1, 0],  # input 1
    [0, 2, 2, 2],  # input 2
    [1, 1, 1, 1],  # input 3
]
W_key = [
    [1, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [1, 0, 1],
]
K = [
    [1, 1, 1],  # Key for input 1
    [4, 2, 2],  # Key for input 2
    [3, 1, 2],  # Key for input 3
]

print(matmul(x, W_key))
