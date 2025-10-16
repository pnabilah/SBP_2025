import numpy as np
from .activations import get_activation, get_activation_derivative
from .losses import *

class Dense:
    def __init__(self, units, activation=None, input_shape=None, initializer=None):
        self.units = units
        self.activation_name = activation
        self.activation = get_activation(self.activation_name)
        self.activation_derivative = get_activation_derivative(self.activation_name)
        self.initializer = initializer

        # initialize weights only if input_shape is known (why only initialize at first layer?)
        if input_shape is not None:
            self._initialize_weights(input_shape, initializer)       # input_shape should be tuple of integers

    def _initialize_weights(self, input_dim, method):
        if method == 'he':
            self.W = np.random.randn(input_dim, self.units) * np.sqrt(2. / input_dim)   # layer1, layer2
        elif method == 'xavier':
            self.W = np.random.randn(input_dim, self.units) * np.sqrt(1. / input_dim)
        elif method == 'lecun':
            self.W = np.random.randn(input_dim, self.units) * np.sqrt(1. / input_dim)
        else:
            self.W = np.random.randn(input_dim, self.units) * 0.01
        self.b = np.zeros((1, self.units))              # 1, layer2
        
    def forward(self, x):
        # Initialize weights if this isn't the first layer
        if not hasattr(self, 'W'):
            self._initialize_weights(x.shape[1], method=self.initializer)
        self.input = x
        self.z_in = np.dot(x, self.W) + self.b
        self.z = self.activation(self.z_in)
        return self.z
    
    def backward(self, da, lr, y_true=None, loss=None):
        if y_true is not None and loss is not None:
            self.loss_fn, self.loss_fn_derivative = get_loss_function(loss)
            dz = self.loss_fn_derivative(self.z, y_true)*self.activation_derivative(self.z_in)
        else:
            dz = da * self.activation_derivative(self.z_in)

        dw = np.outer(self.input, dz)
        db = dz
        da_prev = np.dot(dz, self.W.T)

        self.W -= lr * dw
        self.b -= lr * db
        return da_prev