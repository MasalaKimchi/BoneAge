import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
import argparse
import json
from datetime import datetime

import preprocessing as pp
import modeling_advanced as mod
from modeling import callbacks

def create_dual_input_generator(img_generator, gender_data, batch_size):
    """
    Creates a generator that yields both image and gender data for dual-input models.
    """
    gender_idx = 0
    for img_batch, label_batch in img_generator:
        # Get corresponding gender batch
        current_batch_size = img_batch.shape[0]
        gender_batch = gender_data[gender_idx:gender_idx + current_batch_size]
        gender_idx = (gender_idx + current_batch_size) % len(gender_data)
        
        yield [img_batch, gender_batch], label_batch

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
    return parser.parse_args()

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

    # Data generators
    print('Setting up data generators...')
    train_idg = pp.idg(
        horizontal_flip=True,
        vertical_flip=False,
        height_shift_range=0.2,
        width_shift_range=0.2,
        rotation_range=20,
        shear_range=0.2,
        fill_mode='nearest',
        zoom_range=0.2,
    )
    test_idg = pp.idg()

    train_img_inputs = pp.gen_img_inputs(train_idg, df_train, train_path, batch_size_train, seed, True, 'raw', img_size)
    val_img_inputs = pp.gen_img_inputs(test_idg, df_val, val_path, batch_size_val, seed, False, 'raw', img_size)
    test_img_inputs = pp.gen_img_inputs(test_idg, df_test, test_path, batch_size_test, seed, False, 'raw', img_size)

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

    # Optimizer and callbacks
    optim = mod.optimizer(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0)
    all_callbacks = callbacks(factor=0.8, patience=5, min_lr=1e-6)
    callbacks_no_early = [cb for cb in all_callbacks if cb.__class__.__name__ != 'EarlyStopping']

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

    # Prepare training data based on model type
    if args.model_type in ['sex', 'attn_sex']:
        # Create dual input generators for sex models
        train_data = create_dual_input_generator(train_img_inputs, train_gender, batch_size_train)
        val_data = create_dual_input_generator(val_img_inputs, val_gender, batch_size_val)
    else:
        # Use image-only generators for baseline model
        train_data = train_img_inputs
        val_data = val_img_inputs

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
        
        # Recompile with a lower learning rate (optionally allow a separate arg for this)
        fine_tune_optim = mod.optimizer(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0)
        model.compile(optimizer=fine_tune_optim, loss='mean_absolute_error', metrics=["mae"])
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
    model.load_weights(weights_path)
    
    # Prepare test data based on model type
    if args.model_type in ['sex', 'attn_sex']:
        test_data = create_dual_input_generator(test_img_inputs, test_gender, batch_size_test)
    else:
        test_data = test_img_inputs
    
    test_loss, test_mae = model.evaluate(test_data, steps=step_size_test, verbose=1)
    print(f'Test loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}')
    
    # Get predictions on test set and save to CSV
    print('Generating predictions on test set...')
    test_predictions = model.predict(test_data, steps=step_size_test, verbose=1)
    
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
    
    # Include all arguments as hyperparameters
    hyperparameters = vars(args)
    results = {
        'best_val_mae': float(best_val_mae),
        'best_val_epoch': int(best_epoch),
        'test_mae': float(test_mae),
        'hyperparameters': hyperparameters
    }
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Minimal training results saved to {json_path}')

    # # Check for NaNs/Infs in validation labels
    # print("NaNs in val:", df_val['boneage_zscore'].isna().sum())
    # print("Infs in val:", np.isinf(df_val['boneage_zscore']).sum())
    # print(df_val['boneage_zscore'].describe())

    # # Check validation generator output
    # x_val, y_val = next(iter(val_img_inputs))
    # print("Validation batch shapes:", x_val.shape, y_val.shape)
    # print("Validation labels sample:", y_val[:10]) 