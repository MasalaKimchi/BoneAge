"""
Cyclical Learning Rate utilities for training neural networks.

This module implements the Cyclical Learning Rate callback based on the paper
"Cyclical Learning Rates for Training Neural Networks" by Leslie N. Smith.
"""

import tensorflow as tf
import numpy as np


class CyclicalLearningRate(tf.keras.callbacks.Callback):
    """
    Cyclical Learning Rate callback that cycles the learning rate between two boundaries.
    
    Based on the paper "Cyclical Learning Rates for Training Neural Networks" by Leslie N. Smith.
    Implements triangular policy by default.
    
    Parameters:
    -----------
    base_lr : float
        Initial learning rate (lower boundary)
    max_lr : float
        Maximum learning rate (upper boundary)
    step_size : int
        Number of training iterations per half cycle
    mode : str
        One of 'triangular', 'triangular2', or 'exp_range'
    gamma : float
        Constant in 'exp_range' scaling function: gamma**(cycle iterations)
    """
    
    def __init__(self, base_lr=1e-4, max_lr=1e-2, step_size=2000, mode='triangular', gamma=1.0):
        super(CyclicalLearningRate, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self.clr_iterations = 0
        self.trn_iterations = 0
        self.history = {}
        self._reset()
        
    def _reset(self, new_base_lr=None, new_max_lr=None, new_step_size=None):
        """
        Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0
        
    def clr(self):
        """
        Calculate the learning rate for the current iteration.
        
        Returns:
        --------
        float
            Current learning rate
        """
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
        elif self.mode == 'triangular2':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) / float(2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * (self.gamma ** self.clr_iterations)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        logs = logs or {}
        if self.clr_iterations == 0:
            tf.keras.backend.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            tf.keras.backend.set_value(self.model.optimizer.lr, self.clr())
            
    def on_batch_end(self, epoch, logs=None):
        """Called at the end of each batch."""
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        
        tf.keras.backend.set_value(self.model.optimizer.lr, self.clr())
        
        self.history.setdefault('lr', []).append(tf.keras.backend.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)
        
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v) 