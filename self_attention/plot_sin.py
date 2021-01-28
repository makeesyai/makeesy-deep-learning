import numpy as np
from matplotlib import pyplot as plt

n_tokens = 100
fs = 0.5
x = np.arange(0, n_tokens, 0.1) * fs
pos_enc = np.sin(x)
print(pos_enc.shape)

plt.plot(x, pos_enc)
plt.xlabel('Embedding Dimensions')
plt.ylabel('Token Position')
plt.show()
