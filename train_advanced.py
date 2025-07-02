import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
import argparse

import preprocessing as pp
import modeling_advanced as mod
from modeling import callbacks

def parse_args():
    parser = argparse.ArgumentParser(description='Train advanced baseline model with flexible options.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--backbone', type=str, default='xception', help='Model backbone architecture')
    parser.add_argument('--activation', type=str, default='tanh', help='Activation function for dense layer')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate after pooling')
    parser.add_argument('--dense_units', type=int, default=500, help='Number of units in dense layer')
    parser.add_argument('--weights', type=str, default='imagenet', help='Pretrained weights to use (imagenet or None)')
    parser.add_argument('--epochs_frozen', type=int, default=5, help='Epochs to train with frozen base')
    parser.add_argument('--epochs_finetune', type=int, default=10, help='Epochs to train with unfrozen base')
    parser.add_argument('--batch_size_train', type=int, default=64, help='Batch size for training')
    parser.add_argument('--batch_size_val', type=int, default=32, help='Batch size for validation')
    parser.add_argument('--batch_size_test', type=int, default=None, help='Batch size for test (default: all)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--fine_tune', action='store_true', help='Enable fine-tuning after initial training (default: False)')
    parser.add_argument('--use_clahe_val_test', action='store_true', help='Use CLAHE-enhanced validation and test images (default: False)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Set directories (update as needed)
    train_path = './Data/train/'
    if args.use_clahe_val_test:
        val_path = './Data/validation_CLAHE/'
        test_path = './Data/test_CLAHE/'
        print('Using CLAHE-enhanced validation and test images.')
    else:
        val_path = './Data/validation/'
        test_path = './Data/test/'

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
    batch_size_test = args.batch_size_test if args.batch_size_test is not None else len(df_test)
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

    # Steps per epoch
    step_size_train = len(df_train) // batch_size_train
    step_size_val = len(df_val) // batch_size_val

    # Optimizer and callbacks
    optim = mod.optimizer(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0)
    all_callbacks = callbacks(factor=0.8, patience=5, min_lr=1e-6)
    callbacks_no_early = [cb for cb in all_callbacks if cb.__class__.__name__ != 'EarlyStopping']

    # Build model
    print('Building advanced baseline model...')
    baseline_model = mod.baseline_model(
        img_dims,
        getattr(tf.keras.activations, args.activation) if hasattr(tf.keras.activations, args.activation) else args.activation,
        optim,
        ["mae"],
        backbone=args.backbone,
        weights=args.weights if args.weights != 'None' else None,
        dropout_rate=args.dropout_rate,
        dense_units=args.dense_units
    )

    # Train model (frozen base)
    print('Training baseline model (frozen base)...')
    baseline_history1 = baseline_model.fit(
        train_img_inputs,
        steps_per_epoch=step_size_train,
        validation_data=val_img_inputs,
        validation_steps=step_size_val,
        epochs=args.epochs_frozen,
        callbacks=callbacks_no_early
    )

    # Fine-tune model (unfreeze some layers)
    if args.fine_tune:
        print('Unfreezing last 2 layers of backbone and fine-tuning...')
        mod.unfreeze_backbone_layers(baseline_model, n_layers=2)
        # Recompile with a lower learning rate (optionally allow a separate arg for this)
        fine_tune_optim = mod.optimizer(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0)
        baseline_model.compile(optimizer=fine_tune_optim, loss='mean_absolute_error', metrics=["mae"])
        baseline_history2 = baseline_model.fit(
            train_img_inputs,
            steps_per_epoch=step_size_train,
            validation_data=val_img_inputs,
            validation_steps=step_size_val,
            epochs=args.epochs_finetune,
            callbacks=callbacks_no_early
        )
        mod.plot_history(baseline_history2)

    # Save training history to CSV
    history = baseline_history2.history if args.fine_tune else baseline_history1.history
    with open('baseline_training_log.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch'] + list(history.keys()))
        for i in range(len(history['loss'])):
            writer.writerow([i+1] + [history[k][i] for k in history.keys()])
    print('Training history saved to baseline_training_log.csv')

    # Print best validation MAE
    best_val_mae = min(history['val_mae'])
    best_epoch = history['val_mae'].index(best_val_mae) + 1
    print(f'Best validation MAE: {best_val_mae:.4f} at epoch {best_epoch}')

    # # Check for NaNs/Infs in validation labels
    # print("NaNs in val:", df_val['boneage_zscore'].isna().sum())
    # print("Infs in val:", np.isinf(df_val['boneage_zscore']).sum())
    # print(df_val['boneage_zscore'].describe())

    # # Check validation generator output
    # x_val, y_val = next(iter(val_img_inputs))
    # print("Validation batch shapes:", x_val.shape, y_val.shape)
    # print("Validation labels sample:", y_val[:10]) 