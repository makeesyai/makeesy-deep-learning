import matplotlib.pyplot as plt
import numpy as np

d_pos_vec = 32
max_positions = 100

frequency_0 = 2
position_enc_0 = np.array([
    [pos / np.power(10000, 2 * frequency_0 / d_pos_vec)] for pos in range(0, max_positions)])

frequency_1 = 4
position_enc_2 = np.array([
    [pos / np.power(10000, 2 * frequency_1 / d_pos_vec)] for pos in range(0, max_positions)])

frequency_2 = 6
position_enc_4 = np.array([
    [pos / np.power(10000, 2 * frequency_2 / d_pos_vec)] for pos in range(0, max_positions)])

x = range(0, max_positions)
sine_0 = np.sin(position_enc_0)
sine_2 = np.sin(position_enc_2)
sine_4 = np.sin(position_enc_4)
plt.plot(x, sine_0, label=f"f={frequency_0}")
plt.plot(x, sine_2, label=f"f={frequency_1}")
plt.plot(x, sine_4, label=f"f={frequency_2}")
plt.ylabel('Embedding Dimensions')
plt.xlabel('Token Position')
plt.legend()
plt.show()
