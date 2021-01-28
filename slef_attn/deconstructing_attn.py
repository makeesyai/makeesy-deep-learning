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

w_key = np.array([
  [0, 0, 1],
  [1, 1, 0],
  [0, 1, 0],
  [1, 1, 0]
])
w_query = np.array([
  [1, 0, 1],
  [1, 0, 0],
  [0, 0, 1],
  [0, 1, 1]
])
w_value = np.array([
  [0, 2, 0],
  [0, 3, 0],
  [1, 0, 3],
  [1, 1, 0]
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

# Convert list into numpy array
query = np.stack(query)
key = np.stack(key)
value = np.stack(value)
# print(query)
# print(key)
# print(value)
# exit()
this_query_contextual = []
for i in range(len(x)):
    this_query = query[i]
    relevance = []
    # Compute this_query relevance to all the Keys (keys-row)
    for j in range(len(key)):
        # Calculate inner product in between this_query and each row of the key matrix
        rel_key_j = this_query @ key[j]
        relevance.append(rel_key_j)

    relevance = np.array(relevance)
    # Apply softmax to get probability scores of relevance
    relevance_scores = softmax(relevance, axis=-1)
    # relevance_scores = relevance_scores.round(decimals=1)
    out = 0
    # Each values-row is multiplied with relevance score and added point-wise
    for k in range(len(relevance)):
        # Here value[k] :is vector (of head_dim), and relevance_scores[k]: is a scalar score
        out += value[k] * relevance_scores[k]
    this_query_contextual.append(out.round(decimals=1))

print(np.stack(this_query_contextual))

# For Multi-Head, repeat the above process for n-separate w_query, w_key, and w_value,
# that will be n multi-head attn (In an optimized implementation, all the heads are packed
# in a single matrix for query, key, and value)
