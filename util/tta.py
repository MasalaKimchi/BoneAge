"""
Test Time Augmentation (TTA) utilities for improved predictions.

This module provides Test Time Augmentation functionality that can improve
model predictions by averaging predictions from multiple augmented versions
of the same input.
"""

import numpy as np


def test_time_augmentation(model, test_data, steps, tta_steps=5):
    """
    Perform Test Time Augmentation for better predictions.
    
    Test Time Augmentation involves making predictions on multiple augmented
    versions of the same input and averaging the results. This can improve
    prediction accuracy and robustness.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained model to use for predictions
    test_data : generator
        Test data generator that yields augmented versions
    steps : int
        Number of steps per epoch for the test data
    tta_steps : int
        Number of TTA iterations to perform
        
    Returns:
    --------
    numpy.ndarray
        Averaged predictions from all TTA steps
    """
    predictions_list = []
    
    for i in range(tta_steps):
        print(f'TTA step {i+1}/{tta_steps}')
        predictions = model.predict(test_data, steps=steps, verbose=0)
        predictions_list.append(predictions)
    
    # Average all predictions
    final_predictions = np.mean(predictions_list, axis=0)
    return final_predictions 