"""
Enhanced loss functions for robust regression in bone age prediction.

This module provides various loss functions that can be more robust than standard
MAE/MSE for regression tasks, particularly useful for medical image analysis.
"""

import tensorflow as tf


def huber_loss(delta=1.0):
    """
    Huber loss function for robust regression.
    
    Huber loss combines the best properties of L1 and L2 loss. It is less sensitive
    to outliers than MSE and provides better convergence than MAE.
    
    Parameters:
    -----------
    delta : float
        Threshold parameter that determines the transition point between L1 and L2 loss
        
    Returns:
    --------
    function
        Huber loss function
    """
    def loss(y_true, y_pred):
        residual = tf.abs(y_true - y_pred)
        condition = tf.less(residual, delta)
        small_res = 0.5 * tf.square(residual)
        large_res = delta * residual - 0.5 * tf.square(delta)
        return tf.where(condition, small_res, large_res)
    return loss


def smooth_l1_loss(sigma=1.0):
    """
    Smooth L1 loss function (also known as Huber loss with different parameterization).
    
    This loss function is commonly used in object detection tasks and provides
    smooth gradients for small errors while being robust to outliers.
    
    Parameters:
    -----------
    sigma : float
        Smoothing parameter that controls the transition between L1 and L2 loss
        
    Returns:
    --------
    function
        Smooth L1 loss function
    """
    def loss(y_true, y_pred):
        sigma_squared = sigma ** 2
        regression_diff = y_true - y_pred
        regression_loss = tf.where(
            tf.less(tf.abs(regression_diff), 1.0 / sigma_squared),
            0.5 * sigma_squared * tf.pow(regression_diff, 2),
            tf.abs(regression_diff) - 0.5 / sigma_squared
        )
        return regression_loss
    return loss


def wing_loss(w=10.0, epsilon=2.0):
    """
    Wing loss for robust regression.
    
    Wing loss is designed to handle the challenges of regression tasks where
    the target values can have different scales. It provides better gradients
    for small errors while being robust to large errors.
    
    Parameters:
    -----------
    w : float
        Width parameter that controls the non-linear region
    epsilon : float
        Controls the curvature of the non-linear region
        
    Returns:
    --------
    function
        Wing loss function
    """
    def loss(y_true, y_pred):
        diff = tf.abs(y_true - y_pred)
        C = w - w * tf.math.log(1 + w / epsilon)
        loss_val = tf.where(
            diff < w,
            w * tf.math.log(1 + diff / epsilon),
            diff - C
        )
        return loss_val
    return loss


def get_loss_function(loss_type, **kwargs):
    """
    Get the specified loss function.
    
    Parameters:
    -----------
    loss_type : str
        Type of loss function ('mae', 'mse', 'huber', 'smooth_l1', 'wing')
    **kwargs : dict
        Additional parameters for the specific loss function
        
    Returns:
    --------
    function or str
        Loss function or string identifier for built-in losses
        
    Raises:
    -------
    ValueError
        If loss_type is not recognized
    """
    if loss_type == 'mae':
        return 'mean_absolute_error'
    elif loss_type == 'mse':
        return 'mean_squared_error'
    elif loss_type == 'huber':
        delta = kwargs.get('huber_delta', 1.0)
        return huber_loss(delta=delta)
    elif loss_type == 'smooth_l1':
        sigma = kwargs.get('smooth_l1_sigma', 1.0)
        return smooth_l1_loss(sigma=sigma)
    elif loss_type == 'wing':
        w = kwargs.get('wing_w', 10.0)
        epsilon = kwargs.get('wing_epsilon', 2.0)
        return wing_loss(w=w, epsilon=epsilon)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}") 