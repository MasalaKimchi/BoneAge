import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import csv

import preprocessing as pp
import modeling as mod

# Set directories (update as needed)
train_path = './Data/train/'
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
batch_size_train = 64
batch_size_val = 32
batch_size_test = len(df_test)
seed = 42

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
optim = mod.optimizer(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0)
# Remove early stopping from callbacks
all_callbacks = mod.callbacks(factor=0.8, patience=5, min_lr=1e-6)
callbacks = [cb for cb in all_callbacks if cb.__class__.__name__ != 'EarlyStopping']

# Build model
print('Building baseline model...')
baseline_model = mod.baseline_model(img_dims, tf.keras.activations.tanh, optim, ["mae"])
# baseline_model.summary()

# Train model (frozen base)
print('Training baseline model (frozen base)...')
baseline_history1 = baseline_model.fit(
    train_img_inputs,
    steps_per_epoch=step_size_train,
    validation_data=val_img_inputs,
    validation_steps=step_size_val,
    epochs=5,
    callbacks=callbacks
)

# Fine-tune model (unfreeze some layers)
print('Fine-tuning model...')
baseline_model = mod.fine_tune(model=baseline_model, lr=1e-4, metric=["mae"])
baseline_history2 = baseline_model.fit(
    train_img_inputs,
    steps_per_epoch=step_size_train,
    validation_data=val_img_inputs,
    validation_steps=step_size_val,
    epochs=10,
    callbacks=callbacks
)

# Plot training history
mod.plot_history(baseline_history2)

# Save model
baseline_model.save("baseline_model.h5", overwrite=True, save_format="h5")
print('Model saved as baseline_model.h5')

# Save training history to CSV
history = baseline_history2.history
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

# Check for NaNs/Infs in validation labels
print("NaNs in val:", df_val['boneage_zscore'].isna().sum())
print("Infs in val:", np.isinf(df_val['boneage_zscore']).sum())
print(df_val['boneage_zscore'].describe())

# Check validation generator output
x_val, y_val = next(iter(val_img_inputs))
print("Validation batch shapes:", x_val.shape, y_val.shape)
print("Validation labels sample:", y_val[:10]) 