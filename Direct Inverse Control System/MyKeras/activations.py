import numpy as np

def get_activation(name):
    if name == 'relu':
        return lambda x: np.maximum(0, x)
    elif name == 'sigmoid':
        return lambda x: 1 / (1 + np.exp(-x))
    elif name == 'tanh':
        return np.tanh
    else:
        return lambda x: x

def get_activation_derivative(name):
    if name == 'relu':
        return lambda x: (x > 0).astype(float)
    elif name == 'sigmoid':
        sig = lambda x: 1 / (1 + np.exp(-x))
        return lambda x: sig(x) * (1 - sig(x))
    elif name == 'tanh':
        return lambda x: 1 - np.tanh(x)**2
    else:
        return lambda x: np.ones_like(x)
