from numpy import matmul

x = [
    [1, 0, 1, 0],  # input 1
    [0, 2, 2, 2],  # input 2
    [1, 1, 1, 1],  # input 3
]
W_value = [
    [1, 0, 1],
    [1, 1, 0],
    [0, 1, 1],
    [0, 0, 1],
]
V = [
    [1, 1, 2],  # Value for input 1
    [2, 4, 4],  # Value for input 2
    [2, 2, 3],  # Value for input 3
]

print(matmul(x, W_value))
