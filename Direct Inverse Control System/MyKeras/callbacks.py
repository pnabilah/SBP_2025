import copy
    
class EarlyStopping:
    def __init__(self, patience=50, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_val_loss = float('inf')
        self.patience_count = 0
        self.stop_training = False
        self.best_weights = None

    def on_epoch_end(self, epoch, val_loss, model):
        # Check improvement
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.patience_count = 0

            if self.restore_best_weights:
                self.best_weights = [copy.deepcopy(layer.__dict__) for layer in model.layers]
        else:
            self.patience_count += 1

        # If patience exceeded
        if self.patience_count >= self.patience:
            print(f"Early stopping at epoch {epoch+1}, val_loss={val_loss:.6f}")
            self.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                for layer, best_state in zip(model.layers, self.best_weights):
                    layer.__dict__.update(best_state)
                print("Restored best model weights")