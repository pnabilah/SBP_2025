import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class NN:
    def __init__ (self, input_size, hidden_size, output_size, lr=0.01):
        np.random.seed(42)

        # Inisialisasi parameter jaringan
        self.layer_sizes = [input_size] + hidden_size + [output_size]
        self.lr = lr
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []

        for i in range(len(self.layer_sizes) - 1):
            w = np.random.randn(self.layer_sizes[i+1], self.layer_sizes[i+1]) * np.sqrt(2. / self.layer_sizes[i])
            b = np.zeros((self.layer_sizes[i+1]), 1)
            self.weights.append(w)
            self.biases.append(b)

    # -------------------------------------- Fungsi Aktivasi -------------------------------------- #
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return z * (1 - z)
    
    def reLu(self, z):
        return np.maximum(0, z)
    
    def reLu_derivative(self, z):
        return np.where(z > 0, 1, 0)
    
    def tanh(self, z):
        return np.tanh(z)
    
    def tanh_derivative(self, z):
        return 1 - np.tanh(z) ** 2
    
    # -------------------------------------- Forward Propagation -------------------------------------- #
    def forward(self, X):
        
    
    # -------------------------------------- Backward Propagation -------------------------------------- #
    def backward(self, X, y):
        
