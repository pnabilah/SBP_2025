import numpy as np
import copy

# Fungsi Aktivasi 
# ==========================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def d_sigmoid(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1 - np.tanh(x) ** 2

def reLu(x):
    return np.maximum(0, x)
def d_reLu(x):
    return np.where(x > 0, 1, 0)

ACTIVATIONS = {
    'sigmoid': (sigmoid, d_sigmoid),
    'tanh': (tanh, d_tanh),
    'reLu': (reLu, d_reLu)
}


# Fungsi Loss
# ==========================================
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
def d_mse(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

LOSSES = {
    'mse': (mse, d_mse)
}


# Layer Fully Connected
# ==========================================
class Dense:
    def __init__(self, input_size, output_size, activation='tanh', init_method = 'he'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation, self.d_activation = ACTIVATIONS[activation]
        self.init_method = init_method

        np.random.seed(42)
        self.init_weights_bias()
    
    # ------------------------ Inisialisasi Bobot dan Bias ------------------------ #
    def init_weights_bias(self):
        # Insialisasi bobot
        if self.init_method == 'he':
            self.W = np.random.randn(self.output_size, self.input_size) * np.sqrt(2. / self.input_size)
        elif self.init_method == 'xavier':
            self.W = np.random.randn(self.output_size, self.input_size) * np.sqrt(1. / self.input_size)
        else:
            self.W = np.random.randn(self.output_size, self.input_size) * 0.01
        
        # Insialisasi bias
        self.b = np.zeros(self.output_size)

    # ------------------------------- Feedforward --------------------------------- #
    def feedforward(self, x):
        self.x = x
        self.z_in = np.dot(x, self.W) + self.b
        self.z = self.activation(self.z_in)
        return self.z
    
    # ------------------------------- Backpropagation ----------------------------- #
    def backpropagation(self, da, lr, y_true=None, loss=None):
        # Jika ini layer output, hitung gradien berdasarkan loss
        if y_true is not None and loss is not None:
            loss_fnc, d_loss_fnc = LOSSES[loss]
            dz = d_loss_fnc(self.z, y_true) * self.d_activation(self.z_in)
        else:
            dz = da * self.d_activation(self.z_in)

        dw = np.outer(self.x, dz) # gak tau gimana urutannya
        db = dz
        da_prev = np.dot(dz, elf.W.T, )

        # Update bobot dan bias
        self.W -= lr * dw
        self.b -= lr * db
        return da_prev


# Class Neural Network
# ==========================================
class NN:
    def __init__ (self):
        self.layers = []
        self.compiled = False
        self.stop_training = False
        self.best_weights = None
    
    # ------------------------------- Menambahkan Layer ------------------------------ #
    def add(self, layer):
        self.layers.append(layer)

    # ---------------------------- Membangun Arsitektur ---------------------------- #
    def build_architecture(self, layer_dims, activations, init_methods):
        """
            layer_dims (_type_): [input_dim, hidden1_dim, ..., output_dim]
            activations (_type_): ['tanh', 'reLu', ...]
            init_methods (_type_): ['he', 'xavier', ...]
        """
        if len(layer_dims) - 1 != len(activations) or len(layer_dims) - 1 != len(init_methods):
            raise ValueError("Length of activations and init_methods must be equal to length of layer_dims - 1")
        
        self.layers = []

        for i in range(1, len(layer_dims)-1):
            self.add(Dense(layer_dims[i], layer_dims[i+1], activations[i], init_methods[i]))
        
        print(f"Model architecture built: {len(self.layers)} layers.")
    
    # ------------------------------- Kompilasi Model ------------------------------- #
    def compile(self, loss_fnc_name='mse', lr=0.01):
        self.loss_fnc_name = loss_fnc_name
        self.loss_fnc, self.d_loss_fnc = LOSSES[loss_fnc_name]
        self.lr = lr
        self.compiled = True
    
    # -------------------------------- Train Model ---------------------------------- #
    def fit(self, X_train, T_train, X_val=None, T_val=None, epochs=100, tol=1e-4, callbacks=None):
        if not self.compiled:
            raise Exception("You must compile the model before fitting!")
        
        self.callbacks = callbacks
        
        history = {"train_loss":[], "val_loss":[]}
        
        for epoch in range(epochs):
            # ------------- Training -------------
            train_loss = 0
            for i in range(len(X_train)):
                x = X_train[i]
                t = T_train[i]
                
                # Forward pass
                a = x
                for layer in self.layers:
                    a = layer.feedforward(a)
                y_pred = a  
                train_loss += self.loss_fnc(y_pred, t)              

                # Backward pass
                da = None
                for i, layer in enumerate(reversed(self.layers)):
                    if i == 0:
                        da = layer.backpropagation(da, self.lr, y_true=t, loss=self.loss_fnc_name)
                    else:
                        da = layer.backpropagation(da, self.lr)
            train_loss /= len(X_train)
            history["train_loss"].append(train_loss)
                
            # ------------- Validation -------------
            if X_val is not None and T_val is not None:
                val_loss = 0
                for i in range(len(X_val)):
                    x = X_val[i]
                    t = T_val[i]
                    a = x 
                    for layer in self.layers:
                        a = layer.feedforward(a)
                    val_loss += self.loss_fnc(t, a)
                val_loss /= len(X_val)
                history["val_loss"].append(val_loss)
            
            # ------------- Callbacks -------------
            if callbacks is not None:
                for cb in callbacks:
                    if hasattr(cb, "on_epoch_end"):
                        cb.on_epoch_end(epoch, val_loss, self)
                        if getattr(cb, 'stop_training', False):
                            self.stop_training = True
                            break
            
            if self.stop_training:          # if True, break
                break

            # ---------- Early Stopping based on Tolerance ----------
            if train_loss < tol:
                print(f"Training stopped at epoch {epoch+1} train loss = {train_loss:.6f} has reached tolerance.")
                self.best_weights = [copy.deepcopy(layer.__dict__) for layer in self.layers]
                break

            # ---------- Logging ----------
            if (epoch+1) % 100 == 0 or epoch == 0:
                if val_loss is not None:
                    print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
                else:
                    print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}")
            
        return history
    
    # -------------------------------- Predict ---------------------------------- #
    def predict(self, X):
        y_preds = []
        for i in range(len(X)):
            x = X[i]
            a = x
            for layer in self.layers:
                a = layer.feedforward(a)
            y_preds.append(a)
        return np.array(y_preds)
    
    # ----------------------------- Get Best Weights ------------------------------ #
    def get_best_weights(self):
        if self.callbacks is not None:
            for cb in self.callbacks:
                if hasattr(cb, 'best_weights') and cb.best_weights is not None:
                    return cb.best_weights
        return getattr(self, "best_weights", None)
