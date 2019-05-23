"""
Source:
https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

Modified to save checkpoint with customed name and path
"""
import numpy as np
import torch
import os


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, save_path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.save_path = save_path

    def __call__(self, val_loss, model, save_checkpoint=False):
        if np.isnan(val_loss):
            val_loss = np.Inf
            self.counter = self.patience
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if save_checkpoint:
                self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if save_checkpoint:
                self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):

        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.10f} --> {val_loss:.10f}).  Saving model ...')
            try:
                os.remove(f'{self.save_path}_{self.val_loss_min:.10f}')
            except Exception as e:
                pass
        self.val_loss_min = val_loss
        torch.save(model, f'{self.save_path}_{self.val_loss_min:.10f}')
