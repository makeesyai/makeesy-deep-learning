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
  [1, 0, 1, 0],  # Input 1
  [0, 2, 0, 2],  # Input 2
  [1, 1, 1, 1]   # Input 3
 ])

print(x)
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

# Generate Key, Value, and Key
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

final_out = []
for i in range(len(x)):
    this_query = query[i]
    relevance = []
    # Compute this_query relevance to all the Keys
    for j in range(len(key)):
        rel_key_j = this_query @ key[j]
        relevance.append(rel_key_j)

    relevance = np.array(relevance)
    # Apply softmax to get probability scores of relevance
    relevance_scores = softmax(relevance, axis=-1)
    # relevance_scores = relevance_scores.round(decimals=1)
    out = 0
    for k in range(len(relevance)):
        out += value[k] * relevance_scores[k]
    final_out.append(out.round(decimals=1))
print(np.stack(final_out))
