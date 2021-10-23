from numpy import matmul

x = [
    [1, 0, 1, 0],  # input 1
    [0, 2, 2, 2],  # input 2
    [1, 1, 1, 1],  # input 3
]
W_query = [
    [0, 0, 1],
    [1, 1, 0],
    [0, 1, 0],
    [1, 1, 0],
]
Q = [
    [0, 1, 1],  # Query for input 1
    [4, 6, 0],  # Query for input 2
    [2, 3, 1],  # Query for input 3
]

print(matmul(x, W_query))
