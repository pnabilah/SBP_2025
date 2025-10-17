import numpy as np

def get_loss_function(name):
        if name == 'mean_squared_error':
            return mse, mse_derivative

        elif name == 'binary_crossentropy':
            return binary_crossentropy, binary_crossentropy_derivative

        else:
            raise ValueError(f"Unknown loss function '{name}'")

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size

def binary_crossentropy(y_pred, y_true):
    eps = 1e-9
    return -np.mean(y_true*np.log(y_pred+eps) + (1-y_true)*np.log(1-y_pred+eps))

def binary_crossentropy_derivative(y_pred, y_true):
    eps = 1e-9
    return (-(y_true / (y_pred+eps)) + (1 - y_true) / (1 - y_pred + eps)) / y_true.size

