# Borrowed: https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e

import numpy as np


def get_synthetic_data():
    # Data Generation
    np.random.seed(42)
    x = np.random.rand(100, 1)
    y = 1 + 2 * x + .1 * np.random.randn(100, 1)

    # Shuffles the indices
    idx = np.arange(100)
    np.random.shuffle(idx)

    # Uses first 80 random indices for train
    train_idx = idx[:80]
    # Uses the remaining indices for validation
    val_idx = idx[80:]

    # Generates train and validation sets
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    return x_train, y_train, x_val, y_val
