'''
This module contains functions to:
- Instantiate metrics, parameters, & other tools for modeling
- Build & compile models
'''
import tensorflow as tf
from tensorflow import keras
from keras import Sequential, models, layers, optimizers
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Flatten, Dropout, Conv2D
from keras.applications.xception import Xception
from tensorflow.keras.optimizers.legacy import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

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

def optimizer(lr, beta_1, beta_2, epsilon, decay):
    '''
    Configures parameters & returns optimizer to pass when compiling model
    '''
    optim = tf.keras.optimizers.legacy.Adam(
        learning_rate = lr,
        beta_1 = beta_1,
        beta_2 = beta_2,
        epsilon = epsilon,
        decay = decay,
        clipnorm=1.0
        )
    
    return optim

def baseline_model(img_dims, activation, optim, metric):
    '''
    Builds & compiles baseline model
    Allows for adjusting of activation function
    
    Parameters
    ----------
    img_size: target size of image input
    activation: activation function for first fully connected layer after pooling & dropout
    optim: optimizer to pass when compiling model
    metric: custom error metric

    Returns
    ----------
    Baseline model: pre-trained CNN/convolutional base with overlying fully connected network
    '''
    # Instantiate convolutional base/pre-trained model
    conv_base = tf.keras.applications.Xception(
        include_top = False,    # Remove final densely connected layer
        weights = "imagenet",
        input_shape = img_dims
    )
    conv_base.trainable = False    # Freeze layers of base model initially
     
    # Build models
    model = models.Sequential()    # using functional API
    model.add(conv_base)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    #model.add(Flatten())
    model.add(layers.Dense(500, activation = activation))
    model.add(Dropout(0.25))
    model.add(layers.Dense(1, activation = tf.keras.activations.linear))

    # Compile model
    model.compile(optimizer = optim, loss = 'mean_absolute_error', metrics = metric)
    
    return model

def attn_model(img_dims, optim, metric):
    '''
    Builds & compiles attention mechanism model
    
    *Code adapted from:
    - 'Attention on Pretrained-VGG16 for Bone Age' notebook by K Scott Mader (https://www.kaggle.com/kmader/attention-on-pretrained-vgg16-for-bone-age)
    
    Parameters
    ----------
    img_dims: target size of image input
    optim: optimizer to pass when compiling model
    metric: custom error metric

    Returns
    ----------
    Attention model: pre-trained CNN/convolutional base with attention mechanism
    and final fully connected network
    '''
    # Image input
    input_img = tf.keras.Input(shape = img_dims)

    # Instantiate convolutional base/pre-trained model
    conv_base = tf.keras.applications.Xception(
        input_shape = img_dims,
        weights = 'imagenet',
        include_top = False
        )
    conv_base.trainable = False    # Freeze layers of base model initially

    # Get depth of base model for later application of attention mechanism
    base_depth = conv_base.layers[-1].get_output_shape_at(0)[-1]

    # Extract features from base model
    base_features = conv_base(input_img)
    bn_features = tf.keras.layers.BatchNormalization()(base_features)

    # Attention layer: sequential convolutional layers to extract features
    attn_layer = Conv2D(128, kernel_size = (1,1), padding = 'same',
                        activation= 'relu')(bn_features)
    attn_layer = Conv2D(64, kernel_size = (1,1), padding = 'same',
                        activation = 'relu')(attn_layer)
    attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same',
                        activation = 'relu')(attn_layer)
    attn_layer = tf.keras.layers.LocallyConnected2D(1, kernel_size = (1,1), padding = 'valid',
                        activation = 'tanh')(attn_layer)

    # Apply attention to all features coming out of batch normalization features
    attn_weights = np.ones((1, 1, 1, base_depth))
    conv = Conv2D(base_depth, kernel_size = (1,1), padding = 'same',
                activation = 'linear', use_bias = False, weights = [attn_weights])
    conv.trainable = False    # Freeze weights
    attn_layer = conv(attn_layer)
    mask_features = tf.math.multiply(attn_layer, bn_features)    # Create mask

    # Global average pooling
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)
    gap_layer = tf.keras.layers.Lambda(lambda x: x[0]/x[1])([gap_features, gap_mask])    # Rescale layer after pooling
    gap_layer = Dropout(0.5)(gap_layer)
    gap_layer = Dense(512, activation = 'swish')(gap_layer)
    gap_layer = Dropout(0.2)(gap_layer)

    # Fully connected network
    x = Dense(512, activation = 'relu')(gap_layer)
    x = Dense(512, activation = 'relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation = 'linear')(x)

    # Instantiate & compile model
    model = tf.keras.Model(inputs = input_img, outputs = x)
    model.compile(loss = 'mean_absolute_error', optimizer = optim, metrics = metric)
    
    return model

def sex_model(img_dims, optim, metric):
    # Image input
    input_img = tf.keras.Input(shape = img_dims)

    # Base model/pretrained CNN
    conv_base = tf.keras.applications.Xception(
        input_shape = img_dims,
        weights='imagenet',
        include_top=False
        )
    conv_base.trainable = False    # Freeze convolutional base layer initially

    # Image model
    base_features = conv_base(input_img)
    image = GlobalAveragePooling2D()(base_features)
    image = Dropout(0.5)(image)
    image = Flatten()(image)
    image = Dense(1024, activation = 'tanh')(image)
    image = Dropout(0.2)(image)
    image = Dense(512, activation = 'relu')(image)

    # Gender model
    input_gender = tf.keras.Input(shape = (1,))
    gender = Dense(32, activation = 'relu')(input_gender)

    # Concatenate image & gender models
    features = tf.concat([image, gender], axis=1)

    # Additional dense layers
    combined = Dense(512, activation = 'relu')(features)
    combined = Dense(512, activation = 'relu')(combined)
    combined = Dropout(0.2)(combined)
    combined = Dense(1, activation = 'linear')(combined)

    # Instantiate model
    model = tf.keras.Model(inputs=[input_img, input_gender], outputs=combined)

    # Compile model
    model.compile(loss='mean_absolute_error', optimizer=optim, metrics=metric)
    
    return model

def attn_sex_model(img_dims, optim, metric):
    '''
    Builds & compiles attention mechanism model

    *Code adapted from:
    - 'Attention on Pretrained-VGG16 for Bone Age' notebook by K Scott Mader (https://www.kaggle.com/kmader/attention-on-pretrained-vgg16-for-bone-age)
    - 'KU BDA 2019 boneage project' notebook by Mads Ehrhorn (https://www.kaggle.com/ehrhorn2019/ku-bda-2019-boneage-project)
    *Model architecture also inspired by: https://www.16bit.ai/blog/ml-and-future-of-radiology
    
    Parameters
    ----------
    img_dims: target size of image input
    optim: optimizer to pass when compiling model
    metric: custom error metric

    Returns
    ----------
    Attention-sex model: pre-trained CNN/convolutional base with attention mechaism,
    incorporating sex as feature, and final fully connected network
    '''
    # Image input
    input_img = tf.keras.Input(shape = img_dims)

    # Base model/pretrained CNN
    conv_base = tf.keras.applications.Xception(
        input_shape = img_dims,
        weights = 'imagenet',
        include_top = False
        )
    conv_base.trainable = False    # Freeze base model initially

    # Get depth of base model for later application of attention mechanism
    base_depth = conv_base.layers[-1].get_output_shape_at(0)[-1]

    # Extract features from base model
    base_features = conv_base(input_img)
    bn_features = tf.keras.layers.BatchNormalization()(base_features)

    # Attention layer: sequential convolutional layers to extract features
    attn_layer = Conv2D(128, kernel_size = (1,1), padding = 'same',
                        activation='relu')(bn_features)
    attn_layer = Conv2D(64, kernel_size = (1,1), padding = 'same',
                        activation = 'relu')(attn_layer)
    attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same',
                        activation = 'relu')(attn_layer)
    attn_layer = tf.keras.layers.LocallyConnected2D(1, kernel_size = (1,1), padding = 'valid',
                        activation = 'tanh')(attn_layer)

    # Apply attention to all features coming out of batch normalization features
    attn_weights = np.ones((1, 1, 1, base_depth))
    conv = Conv2D(base_depth, kernel_size = (1,1), padding = 'same',
                activation = 'linear', use_bias = False, weights = [attn_weights])
    conv.trainable = False    # Freeze weights
    attn_layer = conv(attn_layer)
    mask_features = tf.math.multiply(attn_layer, bn_features)    # Create mask

    # Global average pooling
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)
    gap_layer = tf.keras.layers.Lambda(lambda x: x[0]/x[1])([gap_features, gap_mask])    # rescale after pooling
    gap_layer = Dropout(0.5)(gap_layer)
    gap_layer = Dense(512, activation = 'swish')(gap_layer)
    gap_layer = Dropout(0.2)(gap_layer)

    # Gender as a feature: simple MLP model
    input_gender = tf.keras.Input(shape=(1,))    # binary variable
    gender_feature = Dense(16, activation = 'relu')(input_gender)

    # Concatenate image & gender layers
    features = tf.concat([gap_layer, gender_feature], axis = 1)

    # Additional fully connected network through which to feed concatenated networks
    # to try to derive interactions between image features & gender features
    combined = Dense(512, activation = 'relu')(features)
    combined = Dense(512, activation = 'relu')(combined)
    combined = Dropout(0.2)(combined)
    combined = Dense(1, activation = 'linear')(combined)     # 1 output for regression

    # Instantiate & compile model
    model = tf.keras.Model(inputs=[input_img, input_gender], outputs=combined)
    model.compile(loss='mean_absolute_error', optimizer=optim, metrics=metric)

    return model

def callbacks(factor, patience, min_lr):
    '''
    Instantiates EarlyStopping, ModelCheckpoint, ReduceLRonPlateau functions for callbacks
    Allows for adjusting factor, patience, and min_lr parameters of red_lr_plateau function
    
    Parameters
    ----------
    factor
    patience
    min_lr

    Returns
    ----------
    List of callbacks
    '''
    # Early stopping
    early_stopping = EarlyStopping(
        monitor = 'val_loss',
        min_delta = 0,
        patience = 5,
        verbose = 1,
        mode = 'auto'
    )

    # Model checkpoint
    mc = ModelCheckpoint(
        'best_model.h5',
        monitor = 'val_loss',
        mode = 'min',
        save_best_only = True
        )

    # Reduce lr on plateau
    red_lr_plat = ReduceLROnPlateau(
        monitor = 'val_loss',
        factor = factor,
        patience = patience,
        verbose = 1,
        min_delta = 0.0001,
        mode = 'auto',
        cooldown = 5,
        min_lr = min_lr
        )

    callbacks = [early_stopping, mc, red_lr_plat]
    return callbacks

def fine_tune(model, lr, metric):
    '''
    Unfreezes last 2 convolutional blocks of base model and re-compiles model
    Allows for adjusting of learning rate
    
    Parameters
    ----------
    model: previously compiled and trained model
    lr: new learning rate (lower)
    metric: error metric

    Returns
    ----------
    Re-compiled model with partially unfrozen convolutional base
    '''
    conv_base = model
    
    # Unfreeze last 2 convolutional blocks of base model
    conv_base.trainable = True
    for i, layer in enumerate(conv_base.layers):
        if i < 115:
            layer.trainable = False
        else:
            layer.trainable = True

    # Lower learning rate
    optim = tf.keras.optimizers.legacy.Adam(
        learning_rate = lr,
        beta_1 = 0.9,
        beta_2 = 0.999,
        decay = 0
        )

    # Re-compile model
    model.compile(loss = 'mean_absolute_error', optimizer = optim, metrics = metric)

    return model

def plot_history(history):
    '''
    Plots model training history
    '''
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(mae))

    plt.plot(epochs, mae, 'bo', label='Training MAE')
    plt.plot(epochs, val_mae, 'r', label='Validation MAE')
    plt.title('Training and validation MAE')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
