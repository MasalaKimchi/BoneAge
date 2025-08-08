"""
Streamlined Advanced Training Script for Bone Age Prediction

This script uses the new util package for enhanced training capabilities including:
- Cyclical Learning Rate
- Enhanced loss functions (Huber, Smooth L1, Wing)
- Data augmentation (mixup, label noise)
- Test Time Augmentation
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import argparse
import json
from datetime import datetime

import preprocessing_enhanced as pp
import modeling_advanced as mod
from modeling import callbacks

# Import utilities from the new util package
from util import (
    CyclicalLearningRate,
    get_loss_function,
    create_dual_input_generator_enhanced,
    create_enhanced_generator,
    create_fresh_test_generator,
    test_time_augmentation
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Streamlined advanced training with utilities.')
    
    # Basic model parameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--backbone', type=str, default='xception', help='Model backbone')
    parser.add_argument('--activation', type=str, default='tanh', help='Activation function')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--dense_units', type=int, default=500, help='Dense layer units')
    parser.add_argument('--gender_units', type=int, default=32, help='Gender layer units')
    parser.add_argument('--weights', type=str, default='imagenet', help='Pretrained weights')
    
    # Training parameters
    parser.add_argument('--epochs_frozen', type=int, default=5, help='Frozen training epochs')
    parser.add_argument('--epochs_finetune', type=int, default=10, help='Fine-tuning epochs')
    parser.add_argument('--batch_size_train', type=int, default=64, help='Training batch size')
    parser.add_argument('--batch_size_val', type=int, default=32, help='Validation batch size')
    parser.add_argument('--batch_size_test', type=int, default=32, help='Test batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model_type', type=str, default='baseline', 
                        choices=['baseline', 'sex', 'attn_sex'], help='Model type')
    parser.add_argument('--fine_tune', action='store_true', help='Enable fine-tuning')
    parser.add_argument('--no_clahe', action='store_true', help='Disable CLAHE images')

    # Enhanced training features
    parser.add_argument('--use_cyclical_lr', action='store_true', help='Use cyclical learning rate')
    parser.add_argument('--clr_max_lr', type=float, default=1e-2, help='CLR max learning rate')
    parser.add_argument('--clr_step_size', type=int, default=2000, help='CLR step size')
    parser.add_argument('--clr_mode', type=str, default='triangular', 
                        choices=['triangular', 'triangular2', 'exp_range'], help='CLR mode')
    
    # Loss function options
    parser.add_argument('--loss_type', type=str, default='mae', 
                        choices=['mae', 'mse', 'huber', 'smooth_l1', 'wing'], help='Loss function')
    parser.add_argument('--huber_delta', type=float, default=1.0, help='Huber loss delta')
    parser.add_argument('--smooth_l1_sigma', type=float, default=1.0, help='Smooth L1 sigma')
    parser.add_argument('--wing_w', type=float, default=10.0, help='Wing loss w parameter')
    parser.add_argument('--wing_epsilon', type=float, default=2.0, help='Wing loss epsilon')
    
    # Data augmentation
    parser.add_argument('--label_noise_std', type=float, default=0.0, help='Label noise std')
    parser.add_argument('--mixup_alpha', type=float, default=0.0, help='Mixup alpha')
    
    # Regularization
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0, help='Gradient clipping')
    
    # Test Time Augmentation
    parser.add_argument('--use_tta', action='store_true', help='Use Test Time Augmentation')
    parser.add_argument('--tta_steps', type=int, default=5, help='TTA steps')

    return parser.parse_args()


def setup_data_paths(args):
    """Setup data paths based on CLAHE preference."""
    train_path = './Data/train/'
    if args.no_clahe:
        val_path = './Data/validation/'
        test_path = './Data/test/'
        print('Using standard validation and test images.')
    else:
        val_path = './Data/validation_CLAHE/'
        test_path = './Data/test_CLAHE/'
        print('Using CLAHE-enhanced validation and test images.')
    
    return train_path, val_path, test_path


def prepare_data():
    """Load and prepare dataframes."""
    print('Loading dataframes...')
    df_train, df_val, df_test = pp.prep_dfs(
        train_csv='./Data/df_train.csv',
        val_csv='./Data/df_val.csv',
        test_csv='./Data/df_test.csv'
    )

    # Remove NaN cases
    print('Removing NaN cases from dataframes...')
    df_train = df_train.dropna()
    df_val = df_val.dropna()
    df_test = df_test.dropna()

    # Compute normalization stats
    df_train['boneage_zscore'] = (df_train['boneage'] - df_train['boneage'].mean()) / df_train['boneage'].std()
    df_val['boneage_zscore'] = (df_val['boneage'] - df_train['boneage'].mean()) / df_train['boneage'].std()
    df_test['boneage_zscore'] = (df_test['boneage'] - df_train['boneage'].mean()) / df_train['boneage'].std()

    return df_train, df_val, df_test


def setup_data_generators(df_train, df_val, train_path, val_path, args):
    """Setup data generators with enhanced augmentation."""
    print('Setting up enhanced data generators...')
    
    # Enhanced data generators with stronger augmentation
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

    train_img_inputs = pp.gen_img_inputs(train_idg, df_train, train_path, args.batch_size_train, args.seed, True, 'raw', (500, 500))
    val_img_inputs = pp.gen_img_inputs(test_idg, df_val, val_path, args.batch_size_val, args.seed, False, 'raw', (500, 500))

    return train_img_inputs, val_img_inputs


def prepare_gender_data(df_train, df_val, df_test, args):
    """Prepare gender data for dual-input models."""
    if args.model_type in ['sex', 'attn_sex']:
        train_gender = df_train['sex'].values
        val_gender = df_val['sex'].values
        test_gender = df_test['sex'].values
        print(f'Using gender information for {args.model_type} model.')
        return train_gender, val_gender, test_gender
    else:
        return None, None, None


def setup_callbacks(args):
    """Setup training callbacks including cyclical learning rate."""
    all_callbacks = callbacks(factor=0.8, patience=5, min_lr=1e-6)
    callbacks_no_early = [cb for cb in all_callbacks if cb.__class__.__name__ != 'EarlyStopping']

    if args.use_cyclical_lr:
        print(f'Using cyclical learning rate: base_lr={args.lr}, max_lr={args.clr_max_lr}, step_size={args.clr_step_size}, mode={args.clr_mode}')
        clr_callback = CyclicalLearningRate(
            base_lr=args.lr,
            max_lr=args.clr_max_lr,
            step_size=args.clr_step_size,
            mode=args.clr_mode
        )
        callbacks_no_early.append(clr_callback)
        # Remove ReduceLROnPlateau to avoid conflicts
        callbacks_no_early = [cb for cb in callbacks_no_early if cb.__class__.__name__ != 'ReduceLROnPlateau']
        print('Removed ReduceLROnPlateau callback to avoid conflicts with cyclical LR.')
    else:
        print('Using standard learning rate scheduling.')

    return all_callbacks, callbacks_no_early


def build_model(args, img_dims):
    """Build model based on type and parameters."""
    print(f'Building {args.model_type} model...')
    
    # Get loss function
    loss_function = get_loss_function(
        args.loss_type,
        huber_delta=args.huber_delta,
        smooth_l1_sigma=args.smooth_l1_sigma,
        wing_w=args.wing_w,
        wing_epsilon=args.wing_epsilon
    )
    print(f'Using loss function: {args.loss_type}')

    # Build model
    if args.model_type == 'baseline':
        model = mod.baseline_model(
            img_dims,
            getattr(tf.keras.activations, args.activation) if hasattr(tf.keras.activations, args.activation) else args.activation,
            mod.optimizer(lr=args.lr),
            ["mae"],
            backbone=args.backbone,
            weights=args.weights if args.weights != 'None' else None,
            dropout_rate=args.dropout_rate,
            dense_units=args.dense_units
        )
    elif args.model_type == 'sex':
        model = mod.sex_model(
            img_dims,
            mod.optimizer(lr=args.lr),
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
            mod.optimizer(lr=args.lr),
            ["mae"],
            backbone=args.backbone,
            weights=args.weights if args.weights != 'None' else None,
            dropout_rate=args.dropout_rate,
            dense_units=args.dense_units,
            gender_units=args.gender_units
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # Recompile with custom loss and gradient clipping
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

    return model


def train_model(model, train_data, val_data, args, callbacks_no_early, step_size_train, step_size_val):
    """Train the model with frozen and fine-tuning phases."""
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

    if args.fine_tune:
        print('Unfreezing last 2 layers of backbone and fine-tuning...')
        if args.model_type == 'baseline':
            mod.unfreeze_backbone_layers(model, n_layers=2)
        else:
            # For sex/attn_sex models, unfreeze the backbone
            conv_base = None
            for layer in model.layers:
                if hasattr(layer, 'layers') and len(layer.layers) > 100:
                    conv_base = layer
                    break
            if conv_base:
                for layer in conv_base.layers[:-2]:
                    layer.trainable = False
                for layer in conv_base.layers[-2:]:
                    layer.trainable = True
        
        # Recompile with lower learning rate for fine-tuning
        fine_tune_lr = args.lr / 10 if not args.use_cyclical_lr else args.lr
        fine_tune_optim = tf.keras.optimizers.Adam(
            learning_rate=fine_tune_lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=args.gradient_clip_norm
        )
        model.compile(optimizer=fine_tune_optim, loss=get_loss_function(args.loss_type), metrics=["mae"])
        history2 = model.fit(
            train_data,
            steps_per_epoch=step_size_train,
            validation_data=val_data,
            validation_steps=step_size_val,
            epochs=args.epochs_finetune,
            callbacks=callbacks_no_early
        )
        mod.plot_history(history2)

    return history1


def evaluate_model(model, df_test, test_path, test_gender, args, step_size_test, weights_path):
    """Evaluate model on test set and generate predictions."""
    # Load best model weights
    print(f'Loading best model weights from {weights_path} for test evaluation...')
    custom_objects = {}
    if args.loss_type == 'huber':
        from util.loss_functions import huber_loss
        custom_objects['loss'] = huber_loss(args.huber_delta)
    elif args.loss_type == 'smooth_l1':
        from util.loss_functions import smooth_l1_loss
        custom_objects['loss'] = smooth_l1_loss(args.smooth_l1_sigma)
    elif args.loss_type == 'wing':
        from util.loss_functions import wing_loss
        custom_objects['loss'] = wing_loss(args.wing_w, args.wing_epsilon)
    
    model = tf.keras.models.load_model(weights_path, custom_objects=custom_objects)
    
    # Evaluate on test set
    print('Creating fresh generator for test evaluation...')
    test_data_eval = create_fresh_test_generator(
        df_test, test_path, test_gender, args.batch_size_test, args.seed, (500, 500), args.model_type
    )
    test_loss, test_mae = model.evaluate(test_data_eval, steps=step_size_test, verbose=1)
    print(f'Test loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}')
    
    # Generate predictions
    print('Creating fresh generator for test predictions...')
    test_data_pred = create_fresh_test_generator(
        df_test, test_path, test_gender, args.batch_size_test, args.seed, (500, 500), args.model_type
    )
    
    if args.use_tta:
        print(f'Using Test Time Augmentation with {args.tta_steps} steps...')
        test_predictions = test_time_augmentation(model, test_data_pred, step_size_test, args.tta_steps)
    else:
        print('Generating predictions on test set...')
        test_predictions = model.predict(test_data_pred, steps=step_size_test, verbose=1)

    return test_predictions, test_mae


def save_results(df_train, df_test, test_predictions, test_mae, args, weights_path):
    """Save predictions and results."""
    # Convert z-scores back to original bone age scale
    train_mean = df_train['boneage'].mean()
    train_std = df_train['boneage'].std()
    
    true_bone_ages = df_test['boneage'].values
    predicted_bone_ages = (test_predictions.flatten() * train_std) + train_mean
    
    # Create filenames
    backbone_name = args.backbone.lower()
    model_name = args.model_type
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join('backbone_results', f'{model_name}_{backbone_name}_{date_str}_predictions.csv')
    json_path = os.path.join('backbone_results', f'{model_name}_{backbone_name}_{date_str}.json')
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'image_id': df_test['id'].values,
        'true_bone_age': true_bone_ages,
        'predicted_bone_age': predicted_bone_ages,
        'absolute_error': np.abs(true_bone_ages - predicted_bone_ages)
    })
    predictions_df.to_csv(csv_path, index=False)
    print(f'Predictions saved to {csv_path}')
    
    # Calculate performance metrics
    from scipy.stats import pearsonr
    actual_mae = predictions_df['absolute_error'].mean()
    actual_corr, _ = pearsonr(true_bone_ages, predicted_bone_ages)
    
    print(f'VERIFICATION - Actual test performance:')
    print(f'  MAE in months: {actual_mae:.2f}')
    print(f'  Correlation: {actual_corr:.3f}')
    print(f'  Z-score MAE: {test_mae:.4f}')
    
    # Save results
    results = {
        'test_mae_zscore': float(test_mae),
        'test_mae_months': float(actual_mae),
        'test_correlation': float(actual_corr),
        'hyperparameters': vars(args)
    }
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Complete training results saved to {json_path}')


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup
    train_path, val_path, test_path = setup_data_paths(args)
    df_train, df_val, df_test = prepare_data()
    train_img_inputs, val_img_inputs = setup_data_generators(df_train, df_val, train_path, val_path, args)
    train_gender, val_gender, test_gender = prepare_gender_data(df_train, df_val, df_test, args)
    
    # Calculate steps
    step_size_train = len(df_train) // args.batch_size_train
    step_size_val = len(df_val) // args.batch_size_val
    step_size_test = (len(df_test) + args.batch_size_test - 1) // args.batch_size_test
    
    # Setup callbacks
    all_callbacks, callbacks_no_early = setup_callbacks(args)
    
    # Build model
    img_dims = (500, 500, 3)
    model = build_model(args, img_dims)
    
    # Create filename for saving
    backbone_name = args.backbone.lower()
    model_name = args.model_type
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    weights_path = os.path.join('backbone_results', f'{model_name}_{backbone_name}_{date_str}.h5')
    
    # Update ModelCheckpoint callback
    for cb in all_callbacks:
        if cb.__class__.__name__ == 'ModelCheckpoint':
            cb.filepath = weights_path
    for cb in callbacks_no_early:
        if cb.__class__.__name__ == 'ModelCheckpoint':
            cb.filepath = weights_path
    
    # Prepare training data
    if args.model_type in ['sex', 'attn_sex']:
        train_data = create_dual_input_generator_enhanced(
            train_img_inputs, train_gender, args.batch_size_train,
            label_noise_std=args.label_noise_std,
            mixup_alpha=args.mixup_alpha
        )
        val_data = create_dual_input_generator_enhanced(
            val_img_inputs, val_gender, args.batch_size_val
        )
    else:
        train_data = create_enhanced_generator(
            train_img_inputs, args.batch_size_train,
            label_noise_std=args.label_noise_std,
            mixup_alpha=args.mixup_alpha
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
    if args.gradient_clip_norm != 1.0:
        enhancements.append(f"Grad clip: {args.gradient_clip_norm}")
    
    if enhancements:
        print(f"Enhanced training with: {', '.join(enhancements)}")
    
    # Ensure backbone_results directory exists
    os.makedirs('backbone_results', exist_ok=True)
    
    # Train model
    history = train_model(model, train_data, val_data, args, callbacks_no_early, step_size_train, step_size_val)
    
    # Evaluate model
    test_predictions, test_mae = evaluate_model(model, df_test, test_path, test_gender, args, step_size_test, weights_path)
    
    # Save results
    save_results(df_train, df_test, test_predictions, test_mae, args, weights_path)


if __name__ == '__main__':
    main() 