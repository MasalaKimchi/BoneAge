'''
Enhanced preprocessing module for advanced augmentation and generator utilities.
'''
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import preprocess_input
import pandas as pd
import numpy as np
import os
import cv2

def prep_dfs(train_csv='df_train.csv', val_csv='df_val.csv', test_csv='df_test.csv'):
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)
    df_train['sex'] = df_train['gender'].map(lambda x: 1 if x == 'male' else 0)
    df_val['sex'] = df_val['gender'].map(lambda x: 1 if x == 'male' else 0)
    df_test['sex'] = df_test['gender'].map(lambda x: 1 if x == 'male' else 0)
    return df_train, df_val, df_test

def add_gaussian_noise_to_labels(labels, noise_std=0.01):
    import tensorflow as tf
    noise = tf.random.normal(tf.shape(labels), mean=0.0, stddev=noise_std)
    return labels + noise

def mixup(x, y, alpha=0.2):
    import tensorflow as tf
    import numpy as np
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = tf.shape(x[0])[0] if isinstance(x, (list, tuple)) else tf.shape(x)[0]
    index = tf.random.shuffle(tf.range(batch_size))
    if isinstance(x, (list, tuple)):
        mixed_x = [lam * xi + (1 - lam) * tf.gather(xi, index) for xi in x]
    else:
        mixed_x = lam * x + (1 - lam) * tf.gather(x, index)
    mixed_y = lam * y + (1 - lam) * tf.gather(y, index)
    return mixed_x, mixed_y

def create_dual_input_generator_enhanced(img_generator, gender_data, batch_size, label_noise_std=0.0, mixup_alpha=0.0):
    gender_idx = 0
    for img_batch, label_batch in img_generator:
        current_batch_size = img_batch.shape[0]
        gender_batch = gender_data[gender_idx:gender_idx + current_batch_size]
        gender_idx = (gender_idx + current_batch_size) % len(gender_data)
        if label_noise_std > 0:
            label_batch = add_gaussian_noise_to_labels(label_batch, label_noise_std)
        if mixup_alpha > 0:
            img_batch, label_batch = mixup([img_batch, gender_batch], label_batch, mixup_alpha)
            gender_batch = img_batch[1]
            img_batch = img_batch[0]
        yield [img_batch, gender_batch], label_batch

def create_enhanced_generator(img_generator, batch_size, label_noise_std=0.0, mixup_alpha=0.0):
    for img_batch, label_batch in img_generator:
        if label_noise_std > 0:
            label_batch = add_gaussian_noise_to_labels(label_batch, label_noise_std)
        if mixup_alpha > 0:
            img_batch, label_batch = mixup(img_batch, label_batch, mixup_alpha)
        yield img_batch, label_batch

def idg_enhanced(horizontal_flip=False, vertical_flip=False, height_shift_range=0, width_shift_range=0, rotation_range=0, shear_range=0, fill_mode='nearest', zoom_range=0, brightness_range=None, channel_shift_range=None, **kwargs):
    idg_args = dict(
        preprocessing_function=preprocess_input,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        height_shift_range=height_shift_range,
        width_shift_range=width_shift_range,
        rotation_range=rotation_range,
        shear_range=shear_range,
        fill_mode=fill_mode,
        zoom_range=zoom_range
    )
    if brightness_range is not None:
        idg_args['brightness_range'] = brightness_range
    if channel_shift_range is not None:
        idg_args['channel_shift_range'] = channel_shift_range
    idg_args.update(kwargs)
    return ImageDataGenerator(**idg_args)

def gen_img_inputs(idg, df, path, batch_size, seed, shuffle, class_mode, target_size, save_to_dir=None):
    inputs = idg.flow_from_dataframe(
        dataframe = df,
        directory = path,
        x_col = 'filename', y_col = 'boneage_zscore',
        batch_size = batch_size,
        seed = seed,
        shuffle = shuffle,
        class_mode = 'raw',
        target_size = target_size,
        color_mode = 'rgb',
        save_to_dir = save_to_dir,
        )
    return inputs

def gen_img_sex_inputs(idg, df, path, batch_size, seed, shuffle, img_size):
    gen_img = idg.flow_from_dataframe(
        dataframe = df,
        directory = path,
        x_col = 'filename', y_col = 'boneage_zscore',
        batch_size = batch_size,
        seed = seed,
        shuffle = shuffle,
        class_mode = "raw",
        target_size = img_size,
        color_mode = 'rgb')
    gen_sex = idg.flow_from_dataframe(
        dataframe = df,
        directory = path,
        x_col = 'filename', y_col = 'sex',
        batch_size = batch_size,
        seed = seed,
        shuffle = shuffle,
        class_mode = "raw",
        target_size = img_size,
        color_mode = 'rgb')
    while True:
        X1i = gen_img.next()
        X2i = gen_sex.next()
        yield [X1i[0], X2i[1]], X1i[1] 