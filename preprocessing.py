'''
This module contains functions to:
- Preprocess images: enhancement, augmentation
'''
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import preprocess_input
import pandas as pd
import numpy as np
import os
import cv2

# Define parameters for images & models
pixels = 500
img_size = (pixels, pixels)
img_dims = (pixels, pixels, 3)
#batch_size_train = 32
#batch_size_val = 32
#seed = 42
#step_size_train = len(df_train) // batch_size_train
#step_size_val = len(df_val) // batch_size_val

def prep_dfs(train_csv='df_train.csv', val_csv='df_val.csv', test_csv='df_test.csv'):
    '''
    Reads .csv files for training, validation, and test data into pandas dataframes
    '''
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)
    
    # Add encoded sex variable to dataframes
    df_train['sex'] = df_train['gender'].map(lambda x: 1 if x == 'male' else 0)
    df_val['sex'] = df_val['gender'].map(lambda x: 1 if x == 'male' else 0)
    df_test['sex'] = df_test['gender'].map(lambda x: 1 if x == 'male' else 0)
    
    return df_train, df_val, df_test

def prep_sex_dfs():
    '''
    Reads .csv files for training, validation, and test data into pandas dataframes
    '''
    df_train_male = pd.read_csv('df_train_male.csv')
    df_val_male = pd.read_csv('df_val_male.csv')
    df_test_male = pd.read_csv('df_test_male.csv')
    
    df_train_female = pd.read_csv('df_train_female.csv')
    df_val_female = pd.read_csv('df_val_female.csv')
    df_test_female = pd.read_csv('df_test_female.csv')
    
    return df_train_male, df_val_male, df_test_male, df_train_female, df_val_female, df_test_female

# Enhance images
def enhance_img(path, path2, df_train):
    '''
    Reads image and enhances contrast using OpenCV
    
    Parameters
    ----------
    path: directory where images are stored
    path2: directory where enhanced images will be stored
    df_train: training dataframe

    Returns
    ----------
    Contrast-enhanced image
    '''
    # Create list of filenames of training images
    filenames = list(df_train['filename'])

    # Open & enhance images
    for filename in filenames:
        img = cv2.imread(path+filename,0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64,64))
        cl1 = clahe.apply(img)
        cv2.imwrite(path2+filename,cl1)

def boneage_mean_std(df_train, df_val):
    '''
    Calculate boneage mean & standard deviation for specified data
    '''
    boneage_mean = (df_train['boneage'].mean() + df_val['boneage'].mean()) / 2
    boneage_std = (df_train['boneage'].std() + df_val['boneage'].std()) / 2
    
    return boneage_mean, boneage_std

def mae_months(y_true, y_pred):
    '''
    Create custom metric to yield mean absolute error (MAE) in months
    
    Parameters
    ----------
    y_true: actual bone age
    y_pred: predicted bone age

    Returns
    ----------
    Mean absolute error in months
    '''
    return mean_absolute_error((boneage_std*y_true + boneage_mean), (boneage_std*y_pred + boneage_mean))

# Data augmentation
# To increase robustness of dataset (increase dataset size and variability) for regularization
def idg(horizontal_flip=False, vertical_flip=False, height_shift_range=0, width_shift_range=0, rotation_range=0, shear_range=0, fill_mode='nearest', zoom_range=0):
    '''
    Instantiates image data generator for augmentation
    
    Parameters
    ----------
    horizontal_flip: flips images horizontally
    vertical_flip: flips images vertically
    height_shift_range: range to shift image vertically
    width_shift_range: range to shift image horizontally
    rotation_range: range for rotation of image
    shear_range: range for shearing of image
    fill_mode: rule for newly shifted pixels
    zoom_range: range for zooming in on image

    Returns
    ----------
    Image data generator
    '''
    idg = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        horizontal_flip = horizontal_flip,
        vertical_flip = vertical_flip,
        height_shift_range = height_shift_range,
        width_shift_range = width_shift_range,
        rotation_range = rotation_range,
        shear_range = shear_range,
        fill_mode = fill_mode,
        zoom_range = zoom_range
    )
    
    return idg

def gen_img_inputs(idg, df, path, batch_size, seed, shuffle, class_mode, target_size, save_to_dir=None):
    '''
    Generates batches of augmented data for single image input
    
    Parameters
    ----------
    df: pandas dataframe for corresponding dataset
    path: directory where data is stored
    batch_size: size of batch for data augmentation
    seed: for consistency in augmentation
    shuffle: whether or not to shuffle data
    class_mode: channels
    target_size: size for input

    Returns
    ----------
    Batches of augmented image data
    '''
    inputs = idg.flow_from_dataframe(
        dataframe = df,
        directory = path,
        x_col = 'filename', y_col = 'boneage_zscore',
        batch_size = batch_size,
        seed = seed,
        shuffle = shuffle,
        class_mode = 'raw',
        target_size = img_size,
        color_mode = 'rgb',
        save_to_dir = save_to_dir,
        )
    
    return inputs

def gen_img_sex_inputs(idg, df, path, batch_size, seed, shuffle, img_size):
    '''
    Generates batches of augmented data for 2 inputs: image, sex
    In order to map sex data to corresponding image, we need to augment the data together
    
    *Code adapted from:
    - 'KU BDA 2019 boneage project' notebook (https://www.kaggle.com/ehrhorn2019/ku-bda-2019-boneage-project)
    *Model architecture also inspired by: https://www.16bit.ai/blog/ml-and-future-of-radiology
    
    Parameters
    ----------
    df: pandas dataframe for corresponding dataset
    path: directory where data is stored
    batch_size: size of batch for data augmentation
    seed: for consistency in augmentation
    shuffle: whether or not to shuffle data
    class_mode: channels
    target_size: size for input

    Returns
    ----------
    Batches of augmented data for image and sex variables
    '''
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
