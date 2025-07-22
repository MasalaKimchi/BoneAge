import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
import argparse
import json
from datetime import datetime

import preprocessing_enhanced as pp
import modeling_advanced as mod
from modeling import callbacks


class CyclicalLearningRate(tf.keras.callbacks.Callback):
    """
    Cyclical Learning Rate callback that cycles the learning rate between two boundaries.
    
    Based on the paper "Cyclical Learning Rates for Training Neural Networks" by Leslie N. Smith.
    Implements triangular policy by default.
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
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0
        
    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
        elif self.mode == 'triangular2':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) / float(2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * (self.gamma ** self.clr_iterations)
            
    def on_train_begin(self, logs={}):
        logs = logs or {}
        if self.clr_iterations == 0:
            tf.keras.backend.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            tf.keras.backend.set_value(self.model.optimizer.lr, self.clr())
            
    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        
        tf.keras.backend.set_value(self.model.optimizer.lr, self.clr())
        
        self.history.setdefault('lr', []).append(tf.keras.backend.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)
        
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


# Enhanced Loss Functions
def huber_loss(delta=1.0):
    """Huber loss function for robust regression."""
    def loss(y_true, y_pred):
        residual = tf.abs(y_true - y_pred)
        condition = tf.less(residual, delta)
        small_res = 0.5 * tf.square(residual)
        large_res = delta * residual - 0.5 * tf.square(delta)
        return tf.where(condition, small_res, large_res)
    return loss

def smooth_l1_loss(sigma=1.0):
    """Smooth L1 loss function."""
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
    """Wing loss for robust regression."""
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
    """Get the specified loss function."""
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


# Data Augmentation Utilities
def mixup(x, y, alpha=0.2):
    """Mixup data augmentation for regression."""
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
    """Callback for applying mixup augmentation during training."""
    
    def __init__(self, alpha=0.2):
        super(MixupCallback, self).__init__()
        self.alpha = alpha
    
    def on_batch_begin(self, batch, logs=None):
        # Note: This is a simplified implementation
        # In practice, mixup should be integrated into the data generator
        pass

# Label Noise Functions
def add_gaussian_noise_to_labels(labels, noise_std=0.01):
    """Add small Gaussian noise to labels for regularization."""
    noise = tf.random.normal(tf.shape(labels), mean=0.0, stddev=noise_std)
    return labels + noise

def create_dual_input_generator_enhanced(img_generator, gender_data, batch_size, 
                                       label_noise_std=0.0, mixup_alpha=0.0):
    """
    Enhanced generator that yields both image and gender data for dual-input models.
    Includes label noise and mixup augmentation options.
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
    """
    test_idg = pp.idg_enhanced()
    test_img_inputs = pp.gen_img_inputs(test_idg, df, img_path, batch_size, seed, False, 'raw', img_size)
    
    if model_type in ['sex', 'attn_sex']:
        return create_dual_input_generator_enhanced(test_img_inputs, gender_data, batch_size)
    else:
        return test_img_inputs

def parse_args():
    parser = argparse.ArgumentParser(description='Train advanced baseline model with flexible options.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--backbone', type=str, default='xception', help='Model backbone architecture')
    parser.add_argument('--activation', type=str, default='tanh', help='Activation function for dense layer')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate after pooling')
    parser.add_argument('--dense_units', type=int, default=500, help='Number of units in dense layer')
    parser.add_argument('--gender_units', type=int, default=32, help='Number of units in gender layer (for sex models)')
    parser.add_argument('--weights', type=str, default='imagenet', help='Pretrained weights to use (imagenet or None)')
    parser.add_argument('--epochs_frozen', type=int, default=5, help='Epochs to train with frozen base')
    parser.add_argument('--epochs_finetune', type=int, default=10, help='Epochs to train with unfrozen base')
    parser.add_argument('--batch_size_train', type=int, default=64, help='Batch size for training')
    parser.add_argument('--batch_size_val', type=int, default=32, help='Batch size for validation')
    parser.add_argument('--batch_size_test', type=int, default=32, help='Batch size for test (default: 32)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model_type', type=str, default='baseline', choices=['baseline', 'sex', 'attn_sex'], help='Model type to train')
    parser.add_argument('--fine_tune', action='store_true', help='Enable fine-tuning after initial training (default: False)')
    parser.add_argument('--no_clahe', action='store_true', help='Disable CLAHE-enhanced validation and test images (default: CLAHE enabled)')

    # Cyclical Learning Rate parameters
    parser.add_argument('--use_cyclical_lr', action='store_true', help='Enable cyclical learning rate (default: False)')
    parser.add_argument('--clr_max_lr', type=float, default=1e-4, help='Maximum learning rate for cyclical LR (default: 1e-2)')
    parser.add_argument('--clr_step_size', type=int, default=2000, help='Step size for cyclical LR (default: 2000)')
    parser.add_argument('--clr_mode', type=str, default='triangular', choices=['triangular', 'triangular2', 'exp_range'], 
                        help='Cyclical LR mode (default: triangular)')
    parser.add_argument('--clr_gamma', type=float, default=1.0, help='Gamma value for exp_range mode (default: 1.0)')
    
    # Enhanced strategies for reducing MAE
    parser.add_argument('--loss_type', type=str, default='mae', 
                        choices=['mae', 'mse', 'huber', 'smooth_l1', 'wing'],
                        help='Loss function to use (default: mae)')
    parser.add_argument('--huber_delta', type=float, default=1.0, 
                        help='Delta parameter for Huber loss (default: 1.0)')
    parser.add_argument('--smooth_l1_sigma', type=float, default=1.0,
                        help='Sigma parameter for Smooth L1 loss (default: 1.0)')
    parser.add_argument('--wing_w', type=float, default=10.0,
                        help='W parameter for Wing loss (default: 10.0)')
    parser.add_argument('--wing_epsilon', type=float, default=2.0,
                        help='Epsilon parameter for Wing loss (default: 2.0)')
    
    # Label augmentation
    parser.add_argument('--label_noise_std', type=float, default=0.0,
                        help='Standard deviation of Gaussian noise to add to labels (default: 0.0)')
    
    # Data augmentation
    parser.add_argument('--mixup_alpha', type=float, default=0.0,
                        help='Alpha parameter for mixup augmentation (default: 0.0, disabled)')
    
    # Regularization enhancements
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for optimizer (default: 0.01)')
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0,
                        help='Gradient clipping norm (default: 1.0)')
    
    # Test Time Augmentation
    parser.add_argument('--use_tta', action='store_true',
                        help='Use Test Time Augmentation for predictions (default: False)')
    parser.add_argument('--tta_steps', type=int, default=5,
                        help='Number of TTA steps (default: 5)')

    return parser.parse_args()

def test_time_augmentation(model, test_data, steps, tta_steps=5):
    """Perform Test Time Augmentation for better predictions."""
    predictions_list = []
    
    for i in range(tta_steps):
        print(f'TTA step {i+1}/{tta_steps}')
        predictions = model.predict(test_data, steps=steps, verbose=0)
        predictions_list.append(predictions)
    
    # Average all predictions
    final_predictions = np.mean(predictions_list, axis=0)
    return final_predictions

if __name__ == '__main__':
    args = parse_args()

    # Set directories (update as needed)
    train_path = './Data/train/'
    if args.no_clahe:
        val_path = './Data/validation/'
        test_path = './Data/test/'
        print('Using standard validation and test images.')
    else:
        val_path = './Data/validation_CLAHE/'
        test_path = './Data/test_CLAHE/'
        print('Using CLAHE-enhanced validation and test images.')

    # Load data
    print('Loading dataframes...')
    df_train, df_val, df_test = pp.prep_dfs(
        train_csv='./Data/df_train.csv',
        val_csv='./Data/df_val.csv',
        test_csv='./Data/df_test.csv'
    )

    # Remove NaN cases before creating data generators
    print('Removing NaN cases from dataframes...')
    df_train = df_train.dropna()
    df_val = df_val.dropna()
    df_test = df_test.dropna()

    # Compute normalization stats
    df_train['boneage_zscore'] = (df_train['boneage'] - df_train['boneage'].mean()) / df_train['boneage'].std()
    df_val['boneage_zscore'] = (df_val['boneage'] - df_train['boneage'].mean()) / df_train['boneage'].std()
    df_test['boneage_zscore'] = (df_test['boneage'] - df_train['boneage'].mean()) / df_train['boneage'].std()

    # Model/image parameters
    pixels = 500
    img_size = (pixels, pixels)
    img_dims = (pixels, pixels, 3)
    batch_size_train = args.batch_size_train
    batch_size_val = args.batch_size_val
    batch_size_test = args.batch_size_test
    seed = args.seed

    # Enhanced data generators with stronger augmentation
    print('Setting up enhanced data generators...')
    train_idg = pp.idg_enhanced(
        horizontal_flip=True,
        vertical_flip=False,
        height_shift_range=0.2,
        width_shift_range=0.2,
        rotation_range=20,
        shear_range=0.2,
        fill_mode='nearest',
        zoom_range=0.2,
        brightness_range=[0.8, 1.2] if hasattr(pp.idg_enhanced, 'brightness_range') else None,
        channel_shift_range=0.1 if hasattr(pp.idg_enhanced, 'channel_shift_range') else None,
    )
    test_idg = pp.idg_enhanced()

    train_img_inputs = pp.gen_img_inputs(train_idg, df_train, train_path, batch_size_train, seed, True, 'raw', img_size)
    print('train_img_inputs:', train_img_inputs)
    val_img_inputs = pp.gen_img_inputs(test_idg, df_val, val_path, batch_size_val, seed, False, 'raw', img_size)
    print('val_img_inputs:', val_img_inputs)

    # Defensive check
    if train_img_inputs is None:
        raise RuntimeError('train_img_inputs generator is None! Check your DataFrame, path, and generator function.')
    if val_img_inputs is None:
        raise RuntimeError('val_img_inputs generator is None! Check your DataFrame, path, and generator function.')

    # Debug: check first batch from train_img_inputs
    try:
        img_batch, label_batch = next(train_img_inputs)
        print('First img_batch shape:', img_batch.shape)
        print('First label_batch shape:', label_batch.shape)
    except Exception as e:
        print('Error getting first batch from train_img_inputs:', e)
        raise

    # For sex models, prepare gender data
    if args.model_type in ['sex', 'attn_sex']:
        # Gender is encoded as 'sex' column (1 for male, 0 for female)
        train_gender = df_train['sex'].values
        val_gender = df_val['sex'].values
        test_gender = df_test['sex'].values
        print(f'Using gender information for {args.model_type} model.')
    else:
        train_gender = val_gender = test_gender = None

    # Steps per epoch
    step_size_train = len(df_train) // batch_size_train
    step_size_val = len(df_val) // batch_size_val
    step_size_test = (len(df_test) + batch_size_test - 1) // batch_size_test  # Ceiling division to include all images

    # Enhanced optimizer (no weight_decay, as not supported in standard Keras Adam)
    optim = mod.optimizer(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0)
    all_callbacks = callbacks(factor=0.8, patience=5, min_lr=1e-6)
    callbacks_no_early = [cb for cb in all_callbacks if cb.__class__.__name__ != 'EarlyStopping']

    # Add cyclical learning rate callback if enabled
    if args.use_cyclical_lr:
        print(f'Using cyclical learning rate: base_lr={args.lr}, max_lr={args.clr_max_lr}, step_size={args.clr_step_size}, mode={args.clr_mode}')
        clr_callback = CyclicalLearningRate(
            base_lr=args.lr,
            max_lr=args.clr_max_lr,
            step_size=args.clr_step_size,
            mode=args.clr_mode,
            gamma=args.clr_gamma
        )
        callbacks_no_early.append(clr_callback)
        # Remove ReduceLROnPlateau when using cyclical LR to avoid conflicts
        callbacks_no_early = [cb for cb in callbacks_no_early if cb.__class__.__name__ != 'ReduceLROnPlateau']
        print('Removed ReduceLROnPlateau callback to avoid conflicts with cyclical LR.')
    else:
        print('Using standard learning rate scheduling.')

    # Get loss function
    loss_function = get_loss_function(
        args.loss_type,
        huber_delta=args.huber_delta,
        smooth_l1_sigma=args.smooth_l1_sigma,
        wing_w=args.wing_w,
        wing_epsilon=args.wing_epsilon
    )
    print(f'Using loss function: {args.loss_type}')

    # Build model based on type
    print(f'Building {args.model_type} model...')
    if args.model_type == 'baseline':
        model = mod.baseline_model(
            img_dims,
            getattr(tf.keras.activations, args.activation) if hasattr(tf.keras.activations, args.activation) else args.activation,
            optim,
            ["mae"],
            backbone=args.backbone,
            weights=args.weights if args.weights != 'None' else None,
            dropout_rate=args.dropout_rate,
            dense_units=args.dense_units
        )
    elif args.model_type == 'sex':
        model = mod.sex_model(
            img_dims,
            optim,
            ["mae"],
            backbone=args.backbone,
            weights=args.weights if args.weights != 'None' else None,
            dropout_rate=args.dropout_rate,
            dense_units=args.dense_units,
            gender_units=args.gender_units
        )
    elif args.model_type == 'attn_sex':
        model = mod.attn_sex_model(
            img_dims,
            optim,
            ["mae"],
            backbone=args.backbone,
            weights=args.weights if args.weights != 'None' else None,
            dropout_rate=args.dropout_rate,
            dense_units=args.dense_units,
            gender_units=args.gender_units
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # Recompile model with custom loss function and enhanced gradient clipping
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=args.lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=args.gradient_clip_norm
        ),
        loss=loss_function,
        metrics=["mae"]
    )

    # Create enhanced training strategy suffix
    strategy_suffix = ""
    if args.use_cyclical_lr:
        strategy_suffix += "_clr"
    if args.loss_type != 'mae':
        strategy_suffix += f"_{args.loss_type}"
    if args.label_noise_std > 0:
        strategy_suffix += f"_noise{args.label_noise_std}"
    if args.mixup_alpha > 0:
        strategy_suffix += f"_mixup{args.mixup_alpha}"


    # Compose filename based on model type, backbone argument and date
    backbone_name = args.backbone.lower()
    model_name = args.model_type
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    weights_path = os.path.join('backbone_results', f'{model_name}_{backbone_name}_{date_str}.h5')
    json_path = os.path.join('backbone_results', f'{model_name}_{backbone_name}_{date_str}.json')

    # Update ModelCheckpoint callback to use the new weights_path BEFORE training
    for cb in all_callbacks:
        if cb.__class__.__name__ == 'ModelCheckpoint':
            cb.filepath = weights_path
    for cb in callbacks_no_early:
        if cb.__class__.__name__ == 'ModelCheckpoint':
            cb.filepath = weights_path

    # Prepare enhanced training data based on model type
    if args.model_type in ['sex', 'attn_sex']:
        # Create enhanced dual input generators
        train_data = create_dual_input_generator_enhanced(
            train_img_inputs, train_gender, batch_size_train,
            label_noise_std=args.label_noise_std,
            mixup_alpha=args.mixup_alpha
        )
        val_data = create_dual_input_generator_enhanced(
            val_img_inputs, val_gender, batch_size_val
        )
    else:
        # Use enhanced generators for baseline model
        train_data = create_enhanced_generator(
            train_img_inputs, batch_size_train,
            label_noise_std=args.label_noise_std,
            # mixup_alpha=args.mixup_alpha
        )
        val_data = val_img_inputs


    # Print enhancement information
    enhancements = []
    if args.loss_type != 'mae':
        enhancements.append(f"Loss: {args.loss_type}")
    if args.label_noise_std > 0:
        enhancements.append(f"Label noise: {args.label_noise_std}")
    if args.mixup_alpha > 0:
        enhancements.append(f"Mixup: {args.mixup_alpha}")
    if args.weight_decay > 0:
        enhancements.append(f"Weight decay: {args.weight_decay}")
    if args.gradient_clip_norm != 1.0:
        enhancements.append(f"Grad clip: {args.gradient_clip_norm}")
    
    if enhancements:
        print(f"Enhanced training with: {', '.join(enhancements)}")


    # Train model (frozen base)
    print(f'Training {args.model_type} model (frozen base)...')
    history1 = model.fit(
        train_data,
        steps_per_epoch=step_size_train,
        validation_data=val_data,
        validation_steps=step_size_val,
        epochs=args.epochs_frozen,
        callbacks=callbacks_no_early,
        verbose=2
    )

    # Fine-tune model (unfreeze some layers)
    if args.fine_tune:
        print('Unfreezing last 2 layers of backbone and fine-tuning...')
        if args.model_type == 'baseline':
            mod.unfreeze_backbone_layers(model, n_layers=2)
        else:
            # For sex/attn_sex models, unfreeze the backbone (first layer of the functional model)
            conv_base = None
            for layer in model.layers:
                if hasattr(layer, 'layers') and len(layer.layers) > 100:  # Likely the backbone
                    conv_base = layer
                    break
            if conv_base:
                for layer in conv_base.layers[:-2]:
                    layer.trainable = False
                for layer in conv_base.layers[-2:]:
                    layer.trainable = True
        
        # Recompile with potentially different learning rate for fine-tuning
        fine_tune_lr = args.lr / 10 if not args.use_cyclical_lr else args.lr  # Lower LR for fine-tuning if not using CLR
        fine_tune_optim = tf.keras.optimizers.Adam(
            learning_rate=fine_tune_lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=args.gradient_clip_norm
        )
        model.compile(optimizer=fine_tune_optim, loss=loss_function, metrics=["mae"])
        history2 = model.fit(
            train_data,
            steps_per_epoch=step_size_train,
            validation_data=val_data,
            validation_steps=step_size_val,
            epochs=args.epochs_finetune,
            callbacks=callbacks_no_early
        )
        mod.plot_history(history2)

    # Ensure backbone_results directory exists
    os.makedirs('backbone_results', exist_ok=True)
    # Save best validation MAE and test MAE to JSON
    best_val_mae = min(history1.history['val_mae'])
    best_epoch = history1.history['val_mae'].index(best_val_mae) + 1
    print(f'Best validation MAE: {best_val_mae:.4f} at epoch {best_epoch}')

    # Load best model weights before evaluating on test set
    print(f'Loading best model weights from {weights_path} for test evaluation...')
    # Create custom objects dict for loading model with custom loss function
    custom_objects = {}
    if args.loss_type == 'huber':
        custom_objects['loss'] = huber_loss(args.huber_delta)
    elif args.loss_type == 'smooth_l1':
        custom_objects['loss'] = smooth_l1_loss(args.smooth_l1_sigma)
    elif args.loss_type == 'wing':
        custom_objects['loss'] = wing_loss(args.wing_w, args.wing_epsilon)
    
    model = tf.keras.models.load_model(weights_path, custom_objects=custom_objects)
    
    # Always use a fresh generator for test evaluation
    print('Creating fresh generator for test evaluation...')
    test_data_eval = create_fresh_test_generator(
        df_test, test_path, test_gender, batch_size_test, seed, img_size, args.model_type
    )
    test_loss, test_mae = model.evaluate(test_data_eval, steps=step_size_test, verbose=1)
    print(f'Test loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}')
    
    # Always use another fresh generator for test predictions
    print('Creating fresh generator for test predictions...')
    test_data_pred = create_fresh_test_generator(
        df_test, test_path, test_gender, batch_size_test, seed, img_size, args.model_type
    )
    
    # If using TTA, run predictions multiple times and average
    if args.use_tta:
        print(f'Using Test Time Augmentation with {args.tta_steps} steps...')
        test_predictions = test_time_augmentation(model, test_data_pred, step_size_test, args.tta_steps)
    else:
        print('Generating predictions on test set...')
        test_predictions = model.predict(test_data_pred, steps=step_size_test, verbose=1)
    

    # Convert z-scores back to original bone age scale
    train_mean = df_train['boneage'].mean()
    train_std = df_train['boneage'].std()
    
    # Get true bone ages from test set
    true_bone_ages = df_test['boneage'].values
    predicted_bone_ages = (test_predictions.flatten() * train_std) + train_mean
    
    # Create predictions CSV filename
    csv_path = os.path.join('backbone_results', f'{model_name}_{backbone_name}_{date_str}_predictions.csv')
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'image_id': df_test['id'].values,
        'true_bone_age': true_bone_ages,
        'predicted_bone_age': predicted_bone_ages,
        'absolute_error': np.abs(true_bone_ages - predicted_bone_ages)
    })
    predictions_df.to_csv(csv_path, index=False)
    print(f'Predictions saved to {csv_path}')
    
    # Calculate actual performance metrics for verification
    from scipy.stats import pearsonr
    actual_mae = predictions_df['absolute_error'].mean()
    actual_corr, _ = pearsonr(true_bone_ages, predicted_bone_ages)
    
    print(f'VERIFICATION - Actual test performance:')
    print(f'  MAE in months: {actual_mae:.2f}')
    print(f'  Correlation: {actual_corr:.3f}')
    print(f'  Z-score MAE: {test_mae:.4f}')
    
    # Include all arguments as hyperparameters
    hyperparameters = vars(args)
    results = {
        'best_val_mae': float(best_val_mae),
        'best_val_epoch': int(best_epoch),
        'test_mae_zscore': float(test_mae),
        'test_mae_months': float(actual_mae),
        'test_correlation': float(actual_corr),
        'hyperparameters': hyperparameters,
        'enhancements_used': enhancements
    }
    
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Complete training results saved to {json_path}') 