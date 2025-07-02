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

# Supported backbone mapping
BACKBONE_MAP = {
    'xception': keras.applications.Xception,
    'resnet50': keras.applications.ResNet50,
    'resnet101': keras.applications.ResNet101,
    'resnet152': keras.applications.ResNet152,
    'resnet50v2': keras.applications.ResNet50V2,
    'resnet101v2': keras.applications.ResNet101V2,
    'resnet152v2': keras.applications.ResNet152V2,
    'mobilenet': keras.applications.MobileNet,
    'mobilenet_v2': keras.applications.MobileNetV2,
    'mobilenet_v3_small': keras.applications.MobileNetV3Small,
    'mobilenet_v3_large': keras.applications.MobileNetV3Large,
    'efficientnetb0': keras.applications.EfficientNetB0,
    'efficientnetb1': keras.applications.EfficientNetB1,
    'efficientnetb2': keras.applications.EfficientNetB2,
    'efficientnetb3': keras.applications.EfficientNetB3,
    'efficientnetb4': keras.applications.EfficientNetB4,
    'efficientnetb5': keras.applications.EfficientNetB5,
    'efficientnetb6': keras.applications.EfficientNetB6,
    'efficientnetb7': keras.applications.EfficientNetB7,
    'convnext_tiny': keras.applications.ConvNeXtTiny,
    'convnext_small': keras.applications.ConvNeXtSmall,
    'convnext_base': keras.applications.ConvNeXtBase,
    'convnext_large': keras.applications.ConvNeXtLarge,
}

def baseline_model(img_dims, activation, optim, metric, backbone='xception', weights='imagenet', dropout_rate=0.5, dense_units=500):
    """
    Builds & compiles a flexible baseline model with selectable backbone.
    All Dense layers (except backbone) use best-practice weight initialization.

    Parameters
    ----------
    img_dims : tuple
        Input image dimensions (height, width, channels)
    activation : str or callable
        Activation function for the dense layer
    optim : keras optimizer
        Optimizer to use for model compilation
    metric : list
        List of metrics for model evaluation
    backbone : str or callable, default 'xception'
        Backbone architecture to use. Supported: 'xception', 'resnet50', 'mobilenet', 'efficientnetb0', etc.
        If a callable is provided, it will be used directly.
    weights : str or None, default 'imagenet'
        Pretrained weights to use. Set to None for random initialization.
    dropout_rate : float, default 0.5
        Dropout rate after pooling
    dense_units : int, default 500
        Number of units in the dense layer

    Returns
    -------
    model : keras.Model
        Compiled Keras model
    """
    # Select backbone
    if callable(backbone):
        conv_base = backbone(
            include_top=False,
            weights=weights,
            input_shape=img_dims
        )
    elif isinstance(backbone, str):
        backbone = backbone.lower()
        if backbone not in BACKBONE_MAP:
            raise ValueError(f"Unsupported backbone '{backbone}'. Supported: {list(BACKBONE_MAP.keys())}")
        conv_base = BACKBONE_MAP[backbone](
            include_top=False,
            weights=weights,
            input_shape=img_dims
        )
    else:
        raise ValueError("'backbone' must be a string or a callable Keras application.")

    conv_base.trainable = False

    # Choose initializer for Dense layers
    relu_activations = [tf.keras.activations.relu, tf.keras.activations.swish, tf.keras.activations.gelu, tf.keras.activations.selu, tf.keras.activations.elu]
    if activation in relu_activations or (isinstance(activation, str) and activation.lower() in ['relu', 'swish', 'gelu', 'selu', 'elu']):
        dense_initializer = keras.initializers.HeNormal()
    else:
        dense_initializer = keras.initializers.GlorotUniform()
    output_initializer = keras.initializers.GlorotUniform()

    # Build model
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(dense_units, activation=activation, kernel_initializer=dense_initializer))
    model.add(layers.Dropout(dropout_rate/2))
    model.add(layers.Dense(1, activation=tf.keras.activations.linear, kernel_initializer=output_initializer))

    # Compile
    model.compile(optimizer=optim, loss='mean_absolute_error', metrics=metric)
    return model

def unfreeze_backbone_layers(model, n_layers=2):
    """
    Unfreezes the last n_layers of the backbone (first layer) in a Sequential model for fine-tuning.
    Parameters
    ----------
    model : keras.Sequential
        The model containing the backbone as its first layer.
    n_layers : int
        Number of layers from the end of the backbone to unfreeze.
    """
    backbone = model.layers[0]
    if hasattr(backbone, 'layers'):
        for layer in backbone.layers[:-n_layers]:
            layer.trainable = False
        for layer in backbone.layers[-n_layers:]:
            layer.trainable = True
    else:
        # If the backbone does not have sublayers, just unfreeze it
        backbone.trainable = True


def plot_history(history):
    '''
    Plots model training history
    The function generates two plots:
    (1) Training and validation MAE (Mean Absolute Error)
    (2) Training and validation Loss
    These are shown separately to allow for easier comparison of the model's performance on the training and validation sets for both the main metric (MAE) and the loss function.
    '''
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(mae))

    plt.figure()
    plt.plot(epochs, mae, 'bo', label='Training MAE')
    plt.plot(epochs, val_mae, 'r', label='Validation MAE')
    plt.title('Training and validation MAE')
    plt.suptitle('Model Mean Absolute Error (MAE)')
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.suptitle('Model Loss')
    plt.legend()

    plt.show()