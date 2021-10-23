from numpy import transpose

K = [
    [1, 1, 1],  # Key for input 1
    [4, 2, 2],  # Key for input 2
    [3, 1, 2],  # Key for input 3
]

K_transpose = [
    [1, 4, 3],
    [1, 2, 1],
    [1, 2, 2],
]

print(transpose(K))
