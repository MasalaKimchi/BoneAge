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

def sex_model(img_dims, optim, metric, backbone='xception', weights='imagenet', dropout_rate=0.5, dense_units=1024, gender_units=32):
    """
    Builds & compiles a flexible sex model with selectable backbone that incorporates gender information.
    
    Parameters
    ----------
    img_dims : tuple
        Input image dimensions (height, width, channels)
    optim : keras optimizer
        Optimizer to use for model compilation
    metric : list
        List of metrics for model evaluation
    backbone : str or callable, default 'xception'
        Backbone architecture to use. Supported backbones from BACKBONE_MAP.
        If a callable is provided, it will be used directly.
    weights : str or None, default 'imagenet'
        Pretrained weights to use. Set to None for random initialization.
    dropout_rate : float, default 0.5
        Dropout rate after pooling
    dense_units : int, default 1024
        Number of units in the main image dense layer
    gender_units : int, default 32
        Number of units in the gender processing layer

    Returns
    -------
    model : keras.Model
        Compiled Keras model with dual inputs (image and gender)
    """
    # Image input
    input_img = tf.keras.Input(shape=img_dims)

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

    conv_base.trainable = False  # Freeze convolutional base initially

    # Weight initializers
    dense_initializer = keras.initializers.HeNormal()
    tanh_initializer = keras.initializers.GlorotUniform()
    output_initializer = keras.initializers.GlorotUniform()

    # Image model
    base_features = conv_base(input_img)
    image = GlobalAveragePooling2D()(base_features)
    image = Dropout(dropout_rate)(image)
    image = Flatten()(image)
    image = Dense(dense_units, activation='tanh', kernel_initializer=tanh_initializer)(image)
    image = Dropout(dropout_rate * 0.4)(image)
    image = Dense(512, activation='relu', kernel_initializer=dense_initializer)(image)

    # Gender model
    input_gender = tf.keras.Input(shape=(1,))
    gender = Dense(gender_units, activation='relu', kernel_initializer=dense_initializer)(input_gender)

    # Concatenate image & gender models
    features = tf.concat([image, gender], axis=1)

    # Additional dense layers
    combined = Dense(512, activation='relu', kernel_initializer=dense_initializer)(features)
    combined = Dense(512, activation='relu', kernel_initializer=dense_initializer)(combined)
    combined = Dropout(dropout_rate * 0.4)(combined)
    combined = Dense(1, activation='linear', kernel_initializer=output_initializer)(combined)

    # Instantiate model
    model = tf.keras.Model(inputs=[input_img, input_gender], outputs=combined)

    # Compile model
    model.compile(loss='mean_absolute_error', optimizer=optim, metrics=metric)
    
    return model

def attn_sex_model(img_dims, optim, metric, backbone='xception', weights='imagenet', dropout_rate=0.5, dense_units=512, gender_units=16):
    """
    Builds & compiles an attention mechanism model with gender incorporation and selectable backbone.
    
    *Code adapted from:
    - 'Attention on Pretrained-VGG16 for Bone Age' notebook by K Scott Mader (https://www.kaggle.com/kmader/attention-on-pretrained-vgg16-for-bone-age)
    - 'KU BDA 2019 boneage project' notebook by Mads Ehrhorn (https://www.kaggle.com/ehrhorn2019/ku-bda-2019-boneage-project)
    *Model architecture also inspired by: https://www.16bit.ai/blog/ml-and-future-of-radiology
    
    Parameters
    ----------
    img_dims : tuple
        Input image dimensions (height, width, channels)
    optim : keras optimizer
        Optimizer to use for model compilation
    metric : list
        List of metrics for model evaluation
    backbone : str or callable, default 'xception'
        Backbone architecture to use. Supported backbones from BACKBONE_MAP.
        If a callable is provided, it will be used directly.
    weights : str or None, default 'imagenet'
        Pretrained weights to use. Set to None for random initialization.
    dropout_rate : float, default 0.5
        Dropout rate after pooling
    dense_units : int, default 512
        Number of units in the main dense layers
    gender_units : int, default 16
        Number of units in the gender processing layer

    Returns
    -------
    model : keras.Model
        Compiled attention model with dual inputs (image and gender)
    """
    # Image input
    input_img = tf.keras.Input(shape=img_dims)

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

    conv_base.trainable = False  # Freeze base model initially

    # Get depth of base model for later application of attention mechanism
    base_depth = conv_base.layers[-1].get_output_shape_at(0)[-1]

    # Extract features from base model
    base_features = conv_base(input_img)
    bn_features = tf.keras.layers.BatchNormalization()(base_features)

    # Weight initializers
    relu_initializer = keras.initializers.HeNormal()
    swish_initializer = keras.initializers.HeNormal()
    output_initializer = keras.initializers.GlorotUniform()

    # Attention layer: sequential convolutional layers to extract features
    attn_layer = Conv2D(128, kernel_size=(1,1), padding='same',
                        activation='relu', kernel_initializer=relu_initializer)(bn_features)
    attn_layer = Conv2D(64, kernel_size=(1,1), padding='same',
                        activation='relu', kernel_initializer=relu_initializer)(attn_layer)
    attn_layer = Conv2D(16, kernel_size=(1,1), padding='same',
                        activation='relu', kernel_initializer=relu_initializer)(attn_layer)
    attn_layer = tf.keras.layers.LocallyConnected2D(1, kernel_size=(1,1), padding='valid',
                        activation='tanh')(attn_layer)

    # Apply attention to all features coming out of batch normalization features
    attn_weights = np.ones((1, 1, 1, base_depth))
    conv = Conv2D(base_depth, kernel_size=(1,1), padding='same',
                activation='linear', use_bias=False, weights=[attn_weights])
    conv.trainable = False  # Freeze weights
    attn_layer = conv(attn_layer)
    mask_features = tf.math.multiply(attn_layer, bn_features)  # Create mask

    # Global average pooling
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)
    gap_layer = tf.keras.layers.Lambda(lambda x: x[0]/x[1])([gap_features, gap_mask])  # rescale after pooling
    gap_layer = Dropout(dropout_rate)(gap_layer)
    gap_layer = Dense(dense_units, activation='swish', kernel_initializer=swish_initializer)(gap_layer)
    gap_layer = Dropout(dropout_rate * 0.4)(gap_layer)

    # Gender as a feature: simple MLP model
    input_gender = tf.keras.Input(shape=(1,))  # binary variable
    gender_feature = Dense(gender_units, activation='relu', kernel_initializer=relu_initializer)(input_gender)

    # Concatenate image & gender layers
    features = tf.concat([gap_layer, gender_feature], axis=1)

    # Additional fully connected network through which to feed concatenated networks
    # to try to derive interactions between image features & gender features
    combined = Dense(dense_units, activation='relu', kernel_initializer=relu_initializer)(features)
    combined = Dense(dense_units, activation='relu', kernel_initializer=relu_initializer)(combined)
    combined = Dropout(dropout_rate * 0.4)(combined)
    combined = Dense(1, activation='linear', kernel_initializer=output_initializer)(combined)  # 1 output for regression

    # Instantiate & compile model
    model = tf.keras.Model(inputs=[input_img, input_gender], outputs=combined)
    model.compile(loss='mean_absolute_error', optimizer=optim, metrics=metric)

    return model


def attn_sex_model_improved(img_dims, optim, metric, backbone='xception', weights='imagenet', dropout_rate=0.5, dense_units=512, gender_units=16):
    """
    Builds & compiles an improved attention mechanism model with focused innovations for gender incorporation.
    
    This improved version incorporates two key innovations:
    - Cross-Attention between image and clinical features for better feature interaction
    - Gated Feature Fusion for adaptive feature combination
    
    These focused improvements are based on recent research and provide the most impact
    while maintaining computational efficiency and preventing overfitting.
    
    *Code adapted from:
    - 'Attention on Pretrained-VGG16 for Bone Age' notebook by K Scott Mader (https://www.kaggle.com/kmader/attention-on-pretrained-vgg16-for-bone-age)
    - 'KU BDA 2019 boneage project' notebook by Mads Ehrhorn (https://www.kaggle.com/ehrhorn2019/ku-bda-2019-boneage-project)
    *Model architecture also inspired by: https://www.16bit.ai/blog/ml-and-future-of-radiology
    
    Parameters
    ----------
    img_dims : tuple
        Input image dimensions (height, width, channels)
    optim : keras optimizer
        Optimizer to use for model compilation
    metric : list
        List of metrics for model evaluation
    backbone : str or callable, default 'xception'
        Backbone architecture to use. Supported backbones from BACKBONE_MAP.
        If a callable is provided, it will be used directly.
    weights : str or None, default 'imagenet'
        Pretrained weights to use. Set to None for random initialization.
    dropout_rate : float, default 0.5
        Dropout rate after pooling
    dense_units : int, default 512
        Number of units in the main dense layers
    gender_units : int, default 16
        Number of units in the gender processing layer

    Returns
    -------
    model : keras.Model
        Compiled improved attention model with dual inputs (image and gender)
    """
    
    def cross_attention(image_features, clinical_features, num_heads=4):
        """Cross-attention between image and clinical features"""
        # Project clinical features to match image feature dimensions
        clinical_projected = tf.keras.layers.Dense(
            image_features.shape[-1], activation='relu'
        )(clinical_features)
        
        # Get the number of channels for key_dim calculation
        channels = image_features.shape[-1]
        
        # Apply cross-attention: image features attend to clinical features
        cross_attended = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=channels // num_heads
        )(image_features, clinical_projected, clinical_projected)
        
        # Add residual connection and layer normalization
        cross_attended = tf.keras.layers.Add()([image_features, cross_attended])
        cross_attended = tf.keras.layers.LayerNormalization()(cross_attended)
        
        return cross_attended
    
    def gated_fusion(image_features, clinical_features):
        """Gated fusion mechanism for adaptive feature combination"""
        # Project features to same dimension
        image_proj = tf.keras.layers.Dense(clinical_features.shape[-1])(image_features)
        
        # Create gate
        gate = tf.keras.layers.Dense(clinical_features.shape[-1], activation='sigmoid')(
            tf.keras.layers.Concatenate()([image_proj, clinical_features])
        )
        
        # Gated fusion
        fused = gate * image_proj + (1 - gate) * clinical_features
        return fused
    
    # Image input
    input_img = tf.keras.Input(shape=img_dims)

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

    conv_base.trainable = False  # Freeze base model initially

    # Weight initializers
    relu_initializer = keras.initializers.HeNormal()
    swish_initializer = keras.initializers.HeNormal()
    output_initializer = keras.initializers.GlorotUniform()

    # Extract features from base model
    base_features = conv_base(input_img)
    bn_features = tf.keras.layers.BatchNormalization()(base_features)

    # Get depth of base model for later application of attention mechanism
    base_depth = bn_features.shape[-1]

    # Enhanced attention mechanism with multiple convolutional layers
    attn_layer = Conv2D(256, kernel_size=(1,1), padding='same',
                        activation='relu', kernel_initializer=relu_initializer)(bn_features)
    attn_layer = Conv2D(128, kernel_size=(3,3), padding='same',
                        activation='relu', kernel_initializer=relu_initializer)(attn_layer)
    attn_layer = Conv2D(64, kernel_size=(1,1), padding='same',
                        activation='relu', kernel_initializer=relu_initializer)(attn_layer)
    attn_layer = Conv2D(16, kernel_size=(1,1), padding='same',
                        activation='relu', kernel_initializer=relu_initializer)(attn_layer)
    attn_layer = tf.keras.layers.LocallyConnected2D(1, kernel_size=(1,1), padding='valid',
                        activation='tanh')(attn_layer)

    # Apply attention to features
    attn_weights = np.ones((1, 1, 1, base_depth))
    conv = Conv2D(base_depth, kernel_size=(1,1), padding='same',
                activation='linear', use_bias=False, weights=[attn_weights])
    conv.trainable = False  # Freeze weights
    attn_layer = conv(attn_layer)
    mask_features = tf.math.multiply(attn_layer, bn_features)  # Create mask

    # Global average pooling
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)
    gap_layer = tf.keras.layers.Lambda(lambda x: x[0]/x[1])([gap_features, gap_mask])  # rescale after pooling
    gap_layer = Dropout(dropout_rate)(gap_layer)
    gap_layer = Dense(dense_units, activation='swish', kernel_initializer=swish_initializer)(gap_layer)
    gap_layer = Dropout(dropout_rate * 0.4)(gap_layer)

    # Enhanced gender processing
    input_gender = tf.keras.Input(shape=(1,))  # binary variable
    gender_feature = Dense(gender_units * 2, activation='relu', kernel_initializer=relu_initializer)(input_gender)
    gender_feature = Dense(gender_units, activation='relu', kernel_initializer=relu_initializer)(gender_feature)
    gender_feature = Dropout(dropout_rate * 0.3)(gender_feature)

    # INNOVATION 1: Cross-attention between image and clinical features
    # Reshape image features for cross-attention
    image_for_cross = tf.keras.layers.Reshape((1, gap_layer.shape[-1]))(gap_layer)
    clinical_for_cross = tf.keras.layers.Reshape((1, gender_feature.shape[-1]))(gender_feature)
    
    # Apply cross-attention
    cross_attended_image = cross_attention(image_for_cross, clinical_for_cross, num_heads=4)
    cross_attended_image = tf.keras.layers.Flatten()(cross_attended_image)

    # INNOVATION 2: Gated fusion of features
    fused_features = gated_fusion(cross_attended_image, gender_feature)

    # Additional dense layers for final prediction
    combined = Dense(dense_units, activation='relu', kernel_initializer=relu_initializer)(fused_features)
    combined = Dense(dense_units // 2, activation='relu', kernel_initializer=relu_initializer)(combined)
    combined = Dropout(dropout_rate * 0.4)(combined)
    combined = Dense(1, activation='linear', kernel_initializer=output_initializer)(combined)  # 1 output for regression

    # Instantiate & compile model
    model = tf.keras.Model(inputs=[input_img, input_gender], outputs=combined)
    model.compile(loss='mean_absolute_error', optimizer=optim, metrics=metric)

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