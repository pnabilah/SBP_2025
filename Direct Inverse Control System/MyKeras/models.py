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

    def fit(self, X_train, T_train, X_val=None, T_val=None, epochs=10000, tol=1e-3, callbacks=None):
        self.callbacks = callbacks

        if not self.compiled:
            raise Exception("You must compile the model before fitting!")
        
        history = {"train_loss":[], "val_loss":[]}
        
        for epoch in range(epochs):
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

            if train_loss < tol:
                print(f"Training stopped at epoch {epoch+1}, train_loss={train_loss:.6f}")
                self.best_weights = [copy.deepcopy(layer.__dict__) for layer in self.layers]
                break
            
            if (epoch+1) % 10 == 0 or epoch == 0:
                if val_loss is not None:
                    print(f"Epoch {epoch+1}: train_loss={mse:.6f}, val_loss={val_loss:.6f}")
                else:
                    print(f"Epoch {epoch+1}: train_loss={mse:.6f}")

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