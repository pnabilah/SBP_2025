import numpy as np
import copy

class NN:
    def __init__(self, input_size, hidden_size, output_size, lr=0.001, seed=42):

        # Parameters
        self.NI = input_size
        self.NH = hidden_size
        self.NO = output_size
        self.lr = lr

        # Weights
        np.random.seed(seed)
        self.v = np.random.rand(self.NI, self.NH)   # input x hidden
        self.vb = np.random.rand(self.NH)           # 1 x hidden
        self.w = np.random.rand(self.NH, self.NO)   # hidden x output
        self.wb = np.random.rand(self.NO)           # 1 x output

    # --- Set Weights ---
    def set_weights(self, weights):
        """Load weights from dict into the model"""
        self.v = weights["v"].copy()
        self.vb = weights["vb"].copy()
        self.w = weights["w"].copy()
        self.wb = weights["wb"].copy()

    # --- Get Weights ---
    def get_weights(self):
        """Return current weights as a dict"""
        return {
            "v": self.v.copy(),
            "vb": self.vb.copy(),
            "w": self.w.copy(),
            "wb": self.wb.copy()
        }
    
    # --- Activation functions ---
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2
    
    def reLu(x):
        return np.maximum(0, x)

    def d_reLu(x):
        return np.where(x > 0, 1, 0)

    # --- Feedforward ---
    def feedforward(self, x):
        # hidden units
        self.z_in = np.dot(x, self.v) + self.vb
        self.z = self.sigmoid(self.z_in)                # change activation function here

        # output units
        self.y_in = np.dot(self.z, self.w) + self.wb
        #self.y = self.sigmoid(self.y_in)                # change activation function here! self.y is object in this class
        self.y = self.y_in                                                   

        return self.y

    # --- Backpropagation ---
    def backpropagation(self, x, t):
        # error output
        # delta_y = (t - self.y)*self.sigmoid_derivative(self.y_in)  # change activation function here
        delta_y = (t - self.y)                                   
        del_w = self.lr * np.outer(self.z, delta_y)
        del_wb = self.lr * delta_y

        # error hidden
        delta_zin = np.dot(delta_y, self.w.T)
        delta_z = delta_zin * self.sigmoid_derivative(self.z_in)  # change activation function here
        del_v = self.lr * np.outer(x, delta_z)
        del_vb = self.lr * delta_z

        # update bobot
        self.w += del_w
        self.wb += del_wb
        self.v += del_v
        self.vb += del_vb

        return np.mean(delta_y**2)   # return MSE

    # --- Training loop with validation for BPNN---
    def fit_bpnn(self, X_train, T_train, X_val=None, T_val=None,
            epochs=1000, tol=0.001, patience=20):
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_weights = None
        patience_ctr = 0             

        for epoch in range(epochs):
            mse = 0
            # training
            for i in range(len(X_train)):
                x = X_train[i]
                t = T_train[i]
                self.feedforward(x)
                mse += self.backpropagation(x, t)
            mse /= len(X_train)
            history["train_loss"].append(mse)       # mse for every epoch

            # validation
            if X_val is not None and T_val is not None:
                val_loss = 0
                for i in range(len(X_val)):
                    x = X_val[i]
                    t = T_val[i]
                    y = self.feedforward(x)
                    val_loss += np.mean((t - y)**2)
                val_loss /= len(X_val)
                history["val_loss"].append(val_loss)

                # early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = (self.v.copy(), self.vb.copy(), self.w.copy(), self.wb.copy())
                    patience_ctr = 0  # reset patience counter
                else:
                    patience_ctr += 1
                
                # stop training if no improvement for 'patience' epochs
                if patience_ctr >= patience:
                    print(f"Early stopping at epoch {epoch+1}, val_loss={val_loss:.6f}, train_loss={mse:.6f}")
                    if best_weights is not None:
                        self.v, self.vb, self.w, self.wb = best_weights
                    break

            # tolerance check (on training loss)
            if mse < tol:
                print(f"Training stopped at epoch {epoch+1}, train_loss={mse:.6f}")
                if best_weights is not None:
                    self.v, self.vb, self.w, self.wb = best_weights
                break

            if (epoch+1) % 10 == 0 or epoch == 0:
                if val_loss is not None:
                    print(f"Epoch {epoch+1}: train_loss={mse:.6f}, val_loss={val_loss:.6f}")
                else:
                    print(f"Epoch {epoch+1}: train_loss={mse:.6f}")

        return history
    
    # --- Prediction ---
    def predict(self, X_test):
        outputs = []
        for x in X_test:
            outputs.append(self.feedforward(x))
        return np.array(outputs)