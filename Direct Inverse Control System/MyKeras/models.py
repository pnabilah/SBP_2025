from .losses import *
import copy
import numpy as np

class NN:
    def __init__(self):
        self.layers = []                    # store added layers (object) here
        self.compiled = False               # flag to check if compiled
        self.stop_training = False          # flag for early stopping
    
    def add(self, layer):
        self.layers.append(layer)

    def compile(self, optimizer=None, loss='mean_squared_error', lr=0.001):
        self.optimizer = optimizer          # still under development
        self.loss_fn_name = loss
        self.loss_fn, self.loss_fn_derivative = get_loss_function(loss)
        self.lr = lr
        self.compiled = True  

    def fit(self, X_train, T_train, X_val=None, T_val=None, epochs=10000, tol=None, callbacks=None):
        self.callbacks = callbacks

        if not self.compiled:
            raise Exception("You must compile the model before fitting!")
        
        history = {"train_loss":[], "val_loss":[]}
        
        for epoch in range(epochs):
            # training
            train_loss = 0
            for i in range(len(X_train)):
                x = X_train[i]
                t = T_train[i]
                
                # Forward pass
                a = x
                for layer in self.layers:
                    a = layer.forward(a)
                y_pred = a  
                train_loss += self.loss_fn(y_pred, t)              

                # Backward pass
                da = None
                for i, layer in enumerate(reversed(self.layers)):
                    if i == 0:
                        da = layer.backward(da, self.lr, y_true=t, loss=self.loss_fn_name)
                    else:
                        da = layer.backward(da, self.lr)
            train_loss /= len(X_train)
            history["train_loss"].append(train_loss)
                
            # Validation 
            if X_val is not None and T_val is not None:
                val_loss = 0
                for i in range(len(X_val)):
                    x = X_val[i]
                    t = T_val[i]
                    a = x 
                    for layer in self.layers:
                        a = layer.forward(a)
                    val_loss += self.loss_fn(a, t)
                val_loss /= len(X_val)
                history["val_loss"].append(val_loss)

            if callbacks is not None:
                for cb in callbacks:
                    if hasattr(cb, "on_epoch_end"):
                        cb.on_epoch_end(epoch, val_loss, self)
                        if getattr(cb, 'stop_training', False):
                            self.stop_training = True
                            break

            if self.stop_training:          # if True, break
                break

            if tol is not None and train_loss < tol:
                print(f"Training stopped at epoch {epoch+1}, train_loss={train_loss:.6f}")
                self.best_weights = [copy.deepcopy(layer.__dict__) for layer in self.layers]
                break

            if epoch == epochs - 1:
                self.best_weights = [copy.deepcopy(layer.__dict__) for layer in self.layers]

        
            if (epoch+1) % 10 == 0 or epoch == 0:
                if val_loss is not None:
                    print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
                else:
                    print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}")

        return history

    def predict(self, X):
        Y = []
        for x in X:
            a = x
            for layer in self.layers:
                a = layer.forward(a)
            Y.append(a)
        return np.array(Y)

    def get_best_weights(self):
        if self.callbacks is not None:
            for cb in self.callbacks:
                if hasattr(cb, 'best_weights') and cb.best_weights is not None:
                    return cb.best_weights
        return getattr(self, "best_weights", None)
    

class NARX_NN:
    def __init__(self):
        self.layers = []                    # store added layers (object) here
        self.compiled = False               # flag to check if compiled
        self.stop_training = False          # flag for early stopping
    
    def add(self, layer):
        self.layers.append(layer)

    def compile(self, optimizer=None, loss='mean_squared_error', lr=0.001):
        self.optimizer = optimizer          # still under development
        self.loss_fn_name = loss
        self.loss_fn, self.loss_fn_derivative = get_loss_function(loss)
        self.lr = lr
        self.compiled = True  

    def set_weights(self, weights):
        """Set weights from a previous NN model (adjusting input layer if needed)."""
        if len(weights) != len(self.layers):
            print("[WARN] Different number of layers between models, will copy up to min length.")
        
        for i, layer in enumerate(self.layers[:len(weights)]):
            w_prev = weights[i]

            # If shapes match, copy directly
            if hasattr(layer, 'W') and w_prev.get('W') is not None:
                if layer.W.shape == w_prev['W'].shape:
                    layer.W = w_prev['W'].copy()
                    layer.b = w_prev['b'].copy()
                else:
                    # handle mismatch (likely first layer due to y_lags)
                    print(f"[INFO] Adjusting layer {i} weight shape from {w_prev['W'].shape} to {layer.W.shape}")
                    old_W = w_prev['W']
                    new_W = np.zeros_like(layer.W)
                    
                    # reuse old weights as much as possible
                    min_in = min(old_W.shape[0], new_W.shape[0])
                    min_out = min(old_W.shape[1], new_W.shape[1])
                    new_W[:min_in, :min_out] = old_W[:min_in, :min_out]
                    layer.W = new_W
                    layer.b = w_prev['b'][:, :min_out] if w_prev['b'].shape[1] > min_out else w_prev['b']
            
            # Copy over non-weight attributes (activation, etc.)
            for key in ['activation_name', 'activation', 'activation_derivative']:
                if key in w_prev:
                    setattr(layer, key, copy.deepcopy(w_prev[key]))

    def fit(self, X_train, T_train, X_val=None, T_val=None, y_lags=2, epochs=1000, tol=None, callbacks=None):
        self.callbacks = callbacks
        self.y_lags = y_lags

        if not self.compiled:
            raise Exception("You must compile the model before fitting!")
        
        self.NO = self.layers[-1].units
        
        history = {"train_loss":[], "val_loss":[]}
        
        for epoch in range(epochs):
            train_loss = 0
            y_buffer = [np.zeros(self.NO) for _ in range(y_lags)]
            for i in range(len(X_train)):
                x_only = X_train[i]
                x = np.concatenate([x_only] + [yb.flatten() for yb in y_buffer])
                t = T_train[i]
                
                # Forward pass
                a = x
                for layer in self.layers:
                    a = layer.forward(a)
                y_pred = a  
                y_buffer = [y_pred] + y_buffer[:-1]
                train_loss += self.loss_fn(y_pred, t)              

                # Backward pass
                da = None
                for i, layer in enumerate(reversed(self.layers)):
                    if i == 0:
                        da = layer.backward(da, self.lr, y_true=t, loss=self.loss_fn_name)
                    else:
                        da = layer.backward(da, self.lr)
            train_loss /= len(X_train)
            history["train_loss"].append(train_loss)
                
            # Validation 
            if X_val is not None and T_val is not None:
                val_loss = 0
                y_buffer_val = [np.zeros(self.NO) for _ in range(y_lags)]
                for i in range(len(X_val)):
                    x_only = X_val[i]
                    x = np.concatenate([x_only] + [yb.flatten() for yb in y_buffer_val])
                    t = T_val[i]
                    a = x 
                    for layer in self.layers:
                        a = layer.forward(a)
                    y_buffer_val = [a] + y_buffer_val[:-1]
                    val_loss += self.loss_fn(a, t)
                val_loss /= len(X_val)
                history["val_loss"].append(val_loss)

            if callbacks is not None:
                for cb in callbacks:
                    if hasattr(cb, "on_epoch_end"):
                        cb.on_epoch_end(epoch, val_loss, self)
                        if getattr(cb, 'stop_training', False):
                            self.stop_training = True
                            break

            if self.stop_training:          # if True, break
                break

            if tol is not None and train_loss < tol:
                print(f"Training stopped at epoch {epoch+1}, train_loss={train_loss:.6f}")
                self.best_weights = [copy.deepcopy(layer.__dict__) for layer in self.layers]
                break

            if epoch == epochs - 1:
                self.best_weights = [copy.deepcopy(layer.__dict__) for layer in self.layers]
            
            if (epoch+1) % 10 == 0 or epoch == 0:
                if val_loss is not None:
                    print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
                else:
                    print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}")

        return history

    def predict(self, X):
        Y = []
        y_buffer = [np.zeros(self.NO) for _ in range(self.y_lags)]
        for x_only in X:
            a = np.concatenate([x_only] + [yb.flatten() for yb in y_buffer])
            for layer in self.layers:
                a = layer.forward(a)
            Y.append(a)
        return np.array(Y)

    def get_best_weights(self):
        if self.callbacks is not None:
            for cb in self.callbacks:
                if hasattr(cb, 'best_weights') and cb.best_weights is not None:
                    return cb.best_weights
        return getattr(self, "best_weights", None)