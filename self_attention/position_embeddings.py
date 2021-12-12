# NOTE: Sign is with even positions and cosine is at odd positions

import matplotlib.pyplot as plt
import numpy as np

d_model = 32
max_positions = 100

i_2 = 2
position_enc_i_2 = np.array([
    [pos / np.power(10000, 2 * i_2 / d_model)] for pos in range(0, max_positions)])

i_4 = 4
position_enc_i_4 = np.array([
    [pos / np.power(10000, 2 * i_4 / d_model)] for pos in range(0, max_positions)])

i_6 = 6
position_enc_i_6 = np.array([
    [pos / np.power(10000, 2 * i_6 / d_model)] for pos in range(0, max_positions)])

x = range(0, max_positions)
sine_0 = np.sin(position_enc_i_2)
print(sine_0.shape)
exit()
sine_2 = np.sin(position_enc_i_4)
sine_4 = np.sin(position_enc_i_6)
plt.plot(x, sine_0, label=f"i={i_2}")
plt.plot(x, sine_2, label=f"i={i_4}")
plt.plot(x, sine_4, label=f"i={i_6}")
plt.ylabel('Embedding Dimensions')
plt.xlabel('Token Position')
plt.legend()
plt.show()
