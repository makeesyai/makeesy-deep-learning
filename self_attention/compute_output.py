import numpy
from numpy import transpose, matmul
V = numpy.asarray([
    [1, 1, 2],  # Value for input 1
    [2, 4, 4],  # Value for input 2
    [2, 2, 3],  # Value for input 3
])

softmax_attn_score = [
    [0.1, 0.7, 0.2],
    [0., 1., 0.],
    [0., 1., 0.],
]

# V = numpy.asarray([
#     [1., 2., 3.],
#     [2., 8., 0.],
#     [2., 6., 3.]])
# softmax_attn_score = [
#     [0.0, 0.5, 0.5],
#     [0.0, 1.0, 0.0],
#     [0.0, 0.9, 0.1]
# ]

"""
V = [
    [1, 1, 2],  # Value for input 1
    [2, 4, 4],  # Value for input 2
    [2, 2, 3],  # Value for input 3
]

softmax_attn_score = [
    [0.1, 0.7, 0.2],
    [0., 1., 0.],
    [0., 1., 0.],
]

[0.1, 0.1, 0.2] = 0.1 * [1, 1, 2]  # Value for input 1
[1.4, 2.8, 2.8] = 0.7 * [2, 4, 4]  # Value for input 2
[0.4, 0.4, 0.6] = 0.2 * [2, 2, 3]  # Value for input 3
---------------
[1.9, 3.3, 3.6]

[0., 0., 0.] = 0. * [1, 1, 2]  # Value for input 1
[2., 4., 4.] = 1. * [2, 4, 4]  # Value for input 2
[0., 0., 0.] = 0. * [2, 2, 3]  # Value for input 3
-------------
[2., 4., 4.]

[0., 0., 0.] = 0. * [1, 1, 2]  # Value for input 1
[2., 4., 4.] = 1. * [2, 4, 4]  # Value for input 2
[0., 0., 0.] = 0. * [2, 2, 3]  # Value for input 3
-------------
[2., 4., 4.]
output = [
    [1.9, 3.3, 3.6],
    [2., 4., 4.],
    [2., 4., 4.],
]

"""

# o1 = 0.1 * numpy.asarray([1, 1, 2])  # Value for input 1
# o2 = 0.7 * numpy.asarray([2, 4, 4])  # Value for input 2
# o3 = 0.2 * numpy.asarray([2, 2, 3])  # Value for input 3

# weighted_v = numpy.asarray([
#     [0.1, 0.1, 0.2],
#     [1.4, 2.8, 2.8],
#     [0.4, 0.4, 0.6],
# ])

# output = [1.9, 3.3, 3.6]

# print(numpy.sum(weighted_v, axis=0))
# print(o1)
# print(o2)
# print(o3)


v_format = V[:, None]
softmax_attn_score_transpose = transpose(softmax_attn_score)
scores_format = softmax_attn_score_transpose[:, :, None]
print(v_format)
print(scores_format)
weighted_V = numpy.array(v_format * scores_format)
print(weighted_V)
print(weighted_V.sum(axis=0))