import numpy as np
import torch

# Valid AUC
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        # self.val_loss_min = np.Inf
        self.val_AUC_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, valid_auc, model):

        score = -valid_auc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(valid_auc, model)
        elif score > self.best_score + self.delta:  # AUC越高越好
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(valid_auc, model)
            self.counter = 0

    def save_checkpoint(self, valid_auc, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'AUC increased ({self.val_AUC_min:.6f} --> {valid_auc:.6f}).  Saving model ...')
        torch.save(model, self.path)
        self.val_AUC_min = valid_auc