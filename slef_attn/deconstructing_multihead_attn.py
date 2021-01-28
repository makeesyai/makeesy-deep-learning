from scipy.special import softmax
import numpy as np

"""
RNN and LSTM
https://colab.research.google.com/github/mrm8488/shared_colab_notebooks/blob/master/basic_self_attention_.ipynb

Attentions
https://www.youtube.com/watch?v=S27pHKBEp30

Position Embeddings
https://www.youtube.com/watch?v=dichIcUZfOw

@ symbol
https://www.python.org/dev/peps/pep-0465/#semantics

"""
x = np.array([
  [1, 0, 1, 0],   # Input 1
  [0, 2, 0, 2],   # Input 2
  [1, 1, 1, 1],   # Input 3
  [1, 2, 1, 2],   # Input 4
  [2, 2, 2, 2],   # Input 5
 ])
seql, emb = x.shape

w_query = np.array([
    [1, 0, 1, 1, 0, 1],
    [1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 1]
])

w_key = np.array([
    [0, 0, 1, 0, 0, 1],
    [1, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0],
    [1, 1, 0, 1, 1, 0]
])
w_value = np.array([
    [0, 2, 0, 0, 2, 0],
    [0, 3, 0, 0, 3, 0],
    [1, 0, 3, 1, 0, 3],
    [1, 1, 0, 1, 1, 0]
])
key = []
query = []
value = []

# Generate Query, Key, and Value
for i in range(len(x)):
    # The out dim: 1X4 @ 4X3 = 1X3 = array(3)
    query_i = x[i] @ w_query
    key_i = x[i] @ w_key
    value_i = x[i] @ w_value
    query.append(query_i)
    key.append(key_i)
    value.append(value_i)

# print(query)
# print(key)
# print(value)
# exit()
heads = 2
head_dim = 3

# Convert list into numpy array
query = np.stack(query).reshape((seql, heads, head_dim))
key = np.stack(key).reshape((seql, heads, head_dim))
value = np.stack(value).reshape((seql, heads, head_dim))
query = np.transpose(query, (1, 0, 2))
key = np.transpose(key, (1, 0, 2))
value = np.transpose(value, (1, 0, 2))

# Transpose key again to get relevance score per head
key = np.transpose(key, (0, 2, 1))

# Generate the relevance score
relevance = query @ key
# Apply softmax to get probability scores of relevance
relevance_scores = softmax(relevance, axis=-1)
print(relevance_scores.round(decimals=2))
exit()
out = relevance_scores @ value
print(out)
