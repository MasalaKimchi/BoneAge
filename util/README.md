# BoneAge Utilities

This package contains various utilities for training deep learning models for bone age prediction. The utilities are organized into modules for better code organization and reusability.

## Overview

The `util` package provides:
- **Cyclical Learning Rate** callbacks for improved training
- **Enhanced Loss Functions** for robust regression
- **Data Augmentation** techniques including mixup and label noise
- **Test Time Augmentation** for improved predictions

## Modules

### 1. Cyclical Learning Rate (`cyclical_lr.py`)

Implements the Cyclical Learning Rate callback based on the paper "Cyclical Learning Rates for Training Neural Networks" by Leslie N. Smith.

**Key Features:**
- Triangular, triangular2, and exponential range modes
- Configurable base and maximum learning rates
- Step size control for cycle length

**Usage:**
```python
from util import CyclicalLearningRate

clr_callback = CyclicalLearningRate(
    base_lr=1e-4,
    max_lr=1e-2,
    step_size=2000,
    mode='triangular'
)

model.fit(..., callbacks=[clr_callback])
```

### 2. Loss Functions (`loss_functions.py`)

Provides enhanced loss functions that are more robust than standard MAE/MSE for regression tasks.

**Available Loss Functions:**
- **Huber Loss**: Combines L1 and L2 loss properties
- **Smooth L1 Loss**: Smooth gradients with outlier robustness
- **Wing Loss**: Designed for regression with different scales

**Usage:**
```python
from util import get_loss_function

# Get a specific loss function
loss_fn = get_loss_function('huber', huber_delta=1.0)
# or
loss_fn = get_loss_function('wing', wing_w=10.0, wing_epsilon=2.0)

model.compile(optimizer='adam', loss=loss_fn, metrics=['mae'])
```

### 3. Data Augmentation (`augmentation.py`)

Provides data augmentation techniques and enhanced data generators.

**Features:**
- **Mixup**: Combines pairs of samples and labels
- **Label Noise**: Adds Gaussian noise to labels for regularization
- **Enhanced Generators**: Support for both baseline and dual-input models

**Usage:**
```python
from util import (
    create_dual_input_generator_enhanced,
    create_enhanced_generator,
    create_fresh_test_generator
)

# For dual-input models (sex, attn_sex)
train_data = create_dual_input_generator_enhanced(
    img_generator, gender_data, batch_size,
    label_noise_std=0.01,
    mixup_alpha=0.2
)

# For baseline models
train_data = create_enhanced_generator(
    img_generator, batch_size,
    label_noise_std=0.01,
    mixup_alpha=0.2
)

# Fresh test generator (fixes exhaustion bug)
test_data = create_fresh_test_generator(
    df_test, test_path, test_gender, batch_size, 
    seed, img_size, model_type
)
```

### 4. Test Time Augmentation (`tta.py`)

Implements Test Time Augmentation for improved prediction accuracy.

**Usage:**
```python
from util import test_time_augmentation

# Perform TTA with multiple steps
predictions = test_time_augmentation(
    model, test_data, steps=step_size_test, tta_steps=5
)
```

## Integration with Training Scripts

To use these utilities in your training scripts, simply import them:

```python
import util

# Use cyclical learning rate
clr_callback = util.CyclicalLearningRate(...)

# Get enhanced loss function
loss_fn = util.get_loss_function('huber', huber_delta=1.0)

# Create enhanced data generators
train_data = util.create_dual_input_generator_enhanced(...)

# Use TTA for predictions
predictions = util.test_time_augmentation(...)
```

## Benefits

1. **Modularity**: Each utility is self-contained and can be used independently
2. **Reusability**: Utilities can be imported and used across different training scripts
3. **Maintainability**: Easier to maintain and update individual components
4. **Documentation**: Each module is well-documented with examples
5. **Flexibility**: Easy to combine different utilities for custom training strategies

## Best Practices

1. **Cyclical Learning Rate**: Use with care and monitor training curves
2. **Loss Functions**: Experiment with different loss functions for your specific dataset
3. **Data Augmentation**: Start with small augmentation values and increase gradually
4. **Test Time Augmentation**: Use for final predictions to improve accuracy

## Dependencies

- TensorFlow 2.x
- NumPy
- Preprocessing modules from the main project

## Examples

See the updated `train_advanced_FIXED_enhanced.py` script for complete examples of how to use these utilities in a full training pipeline. 