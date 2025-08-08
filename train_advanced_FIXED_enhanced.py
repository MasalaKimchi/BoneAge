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

# Import utilities from the new util package
from util import (
    CyclicalLearningRate,
    get_loss_function,
    create_dual_input_generator_enhanced,
    create_enhanced_generator,
    create_fresh_test_generator,
    test_time_augmentation
)


# CyclicalLearningRate class moved to util.cyclical_lr module


# Enhanced loss functions moved to util.loss_functions module


# Data augmentation utilities moved to util.augmentation module

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

# Test Time Augmentation function moved to util.tta module

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