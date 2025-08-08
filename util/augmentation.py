"""
Data augmentation utilities for bone age prediction.

This module provides various data augmentation techniques including mixup,
label noise injection, and enhanced data generators for both baseline
and dual-input models.
"""

import tensorflow as tf
import numpy as np
import preprocessing_enhanced as pp


def mixup(x, y, alpha=0.2):
    """
    Mixup data augmentation for regression.
    
    Mixup creates new training samples by combining pairs of samples and their labels.
    This technique has been shown to improve generalization and robustness.
    
    Parameters:
    -----------
    x : tensor
        Input features (images)
    y : tensor
        Target labels
    alpha : float
        Alpha parameter for beta distribution
        
    Returns:
    --------
    tuple
        Mixed input features and labels
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = tf.shape(x)[0]
    index = tf.random.shuffle(tf.range(batch_size))
    
    mixed_x = lam * x + (1 - lam) * tf.gather(x, index)
    mixed_y = lam * y + (1 - lam) * tf.gather(y, index)
    
    return mixed_x, mixed_y


class MixupCallback(tf.keras.callbacks.Callback):
    """
    Callback for applying mixup augmentation during training.
    
    Note: This is a simplified implementation. In practice, mixup should be
    integrated into the data generator for better performance.
    """
    
    def __init__(self, alpha=0.2):
        super(MixupCallback, self).__init__()
        self.alpha = alpha
    
    def on_batch_begin(self, batch, logs=None):
        # Note: This is a simplified implementation
        # In practice, mixup should be integrated into the data generator
        pass


def add_gaussian_noise_to_labels(labels, noise_std=0.01):
    """
    Add small Gaussian noise to labels for regularization.
    
    Adding noise to labels can help prevent overfitting and improve
    model robustness by making the training process more challenging.
    
    Parameters:
    -----------
    labels : tensor
        Target labels
    noise_std : float
        Standard deviation of the Gaussian noise
        
    Returns:
    --------
    tensor
        Labels with added noise
    """
    noise = tf.random.normal(tf.shape(labels), mean=0.0, stddev=noise_std)
    return labels + noise


def create_dual_input_generator_enhanced(img_generator, gender_data, batch_size, 
                                       label_noise_std=0.0, mixup_alpha=0.0):
    """
    Enhanced generator that yields both image and gender data for dual-input models.
    Includes label noise and mixup augmentation options.
    
    Parameters:
    -----------
    img_generator : generator
        Image data generator
    gender_data : array
        Gender data array
    batch_size : int
        Batch size
    label_noise_std : float
        Standard deviation for label noise (0.0 to disable)
    mixup_alpha : float
        Alpha parameter for mixup (0.0 to disable)
        
    Yields:
    -------
    tuple
        (inputs, labels) where inputs is [images, gender]
    """
    gender_idx = 0
    for img_batch, label_batch in img_generator:
        # Get corresponding gender batch
        current_batch_size = img_batch.shape[0]
        gender_batch = gender_data[gender_idx:gender_idx + current_batch_size]
        gender_idx = (gender_idx + current_batch_size) % len(gender_data)
        
        # Add label noise if specified
        if label_noise_std > 0:
            label_batch = add_gaussian_noise_to_labels(label_batch, label_noise_std)
        
        # Apply mixup if specified
        if mixup_alpha > 0:
            img_batch, label_batch = mixup([img_batch, gender_batch], label_batch, mixup_alpha)
            gender_batch = img_batch[1]
            img_batch = img_batch[0]
        
        yield [img_batch, gender_batch], label_batch


def create_enhanced_generator(img_generator, batch_size, label_noise_std=0.0, mixup_alpha=0.0):
    """
    Enhanced generator for baseline models with augmentation options.
    
    Parameters:
    -----------
    img_generator : generator
        Image data generator
    batch_size : int
        Batch size
    label_noise_std : float
        Standard deviation for label noise (0.0 to disable)
    mixup_alpha : float
        Alpha parameter for mixup (0.0 to disable)
        
    Yields:
    -------
    tuple
        (images, labels)
    """
    for img_batch, label_batch in img_generator:
        # Add label noise if specified
        if label_noise_std > 0:
            label_batch = add_gaussian_noise_to_labels(label_batch, label_noise_std)
        
        # Apply mixup if specified
        if mixup_alpha > 0:
            img_batch, label_batch = mixup(img_batch, label_batch, mixup_alpha)
        
        yield img_batch, label_batch


def create_fresh_test_generator(df, img_path, gender_data, batch_size, seed, img_size, model_type):
    """
    Create a fresh test generator - fixes the generator exhaustion bug.
    
    Parameters:
    -----------
    df : DataFrame
        Test data DataFrame
    img_path : str
        Path to test images
    gender_data : array
        Gender data array (can be None for baseline models)
    batch_size : int
        Batch size
    seed : int
        Random seed
    img_size : tuple
        Image dimensions
    model_type : str
        Type of model ('baseline', 'sex', 'attn_sex')
        
    Returns:
    --------
    generator
        Fresh test data generator
    """
    test_idg = pp.idg_enhanced()
    test_img_inputs = pp.gen_img_inputs(test_idg, df, img_path, batch_size, seed, False, 'raw', img_size)
    
    if model_type in ['sex', 'attn_sex']:
        return create_dual_input_generator_enhanced(test_img_inputs, gender_data, batch_size)
    else:
        return test_img_inputs 