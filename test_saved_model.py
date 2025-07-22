'''
Script to load and test a previously saved model without retraining.
'''
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime
from scipy.stats import pearsonr
import preprocessing as pp

def huber_loss(delta=1.0):
    def loss(y_true, y_pred):
        error = y_true - y_pred
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, delta)
        linear = abs_error - quadratic
        return tf.reduce_mean(0.5 * quadratic**2 + delta * linear)
    return loss

def smooth_l1_loss(sigma=1.0):
    def loss(y_true, y_pred):
        error = y_true - y_pred
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, sigma)
        linear = abs_error - quadratic
        return tf.reduce_mean(0.5 * quadratic**2 + sigma * linear)
    return loss

def wing_loss(w=10.0, epsilon=2.0):
    def loss(y_true, y_pred):
        error = y_true - y_pred
        abs_error = tf.abs(error)
        C = w - w * tf.math.log(1 + w / epsilon)
        wing_loss = tf.where(
            abs_error < w,
            w * tf.math.log(1 + abs_error / epsilon),
            abs_error - C
        )
        return tf.reduce_mean(wing_loss)
    return loss

def create_fresh_test_generator(df, img_path, gender_data, batch_size, seed, img_size, model_type):
    """Create a fresh generator for testing to avoid exhaustion issues."""
    if model_type in ['sex', 'attn_sex']:
        # Create fresh generator for dual input models
        test_idg = pp.idg()
        test_img_inputs = pp.gen_img_inputs(test_idg, df, img_path, batch_size, seed, False, 'raw', img_size)
        # Create a proper dual input generator
        gender_idx = 0
        for img_batch, label_batch in test_img_inputs:
            current_batch_size = img_batch.shape[0]
            gender_batch = gender_data[gender_idx:gender_idx + current_batch_size]
            gender_idx = (gender_idx + current_batch_size) % len(gender_data)
            yield [img_batch, gender_batch], label_batch
    else:
        # Create fresh generator for baseline models
        test_idg = pp.idg()
        test_data = pp.gen_img_inputs(test_idg, df, img_path, batch_size, seed, False, 'raw', img_size)
        for img_batch, label_batch in test_data:
            yield img_batch, label_batch

def test_time_augmentation(model, test_data, steps, tta_steps=5):
    """Perform test time augmentation by averaging multiple predictions."""
    predictions = []
    for _ in range(tta_steps):
        pred = model.predict(test_data, steps=steps, verbose=0)
        predictions.append(pred)
    return np.mean(predictions, axis=0)

def parse_args():
    parser = argparse.ArgumentParser(description='Test a saved model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the saved model (.h5 file)')
    parser.add_argument('--model_type', type=str, default='attn_sex',
                       choices=['baseline', 'sex', 'attn_sex'],
                       help='Type of model architecture')
    parser.add_argument('--loss_type', type=str, default='mae',
                       choices=['mae', 'huber', 'smooth_l1', 'wing'],
                       help='Loss function used during training')
    parser.add_argument('--huber_delta', type=float, default=1.0,
                       help='Delta parameter for Huber loss')
    parser.add_argument('--smooth_l1_sigma', type=float, default=1.0,
                       help='Sigma parameter for Smooth L1 loss')
    parser.add_argument('--wing_w', type=float, default=10.0,
                       help='W parameter for Wing loss')
    parser.add_argument('--wing_epsilon', type=float, default=2.0,
                       help='Epsilon parameter for Wing loss')
    parser.add_argument('--batch_size_test', type=int, default=32,
                       help='Batch size for testing')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--use_tta', action='store_true',
                       help='Use test time augmentation')
    parser.add_argument('--tta_steps', type=int, default=5,
                       help='Number of TTA steps')
    return parser.parse_args()

def main():
    args = parse_args()
        # Set directories (update as needed)
    train_path = './Data/train/'
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

    # Remove NaN cases
    print('Removing NaN cases from dataframes...')
    df_train = df_train.dropna()
    df_val = df_val.dropna()
    df_test = df_test.dropna()
    
    # Calculate bone age statistics
    boneage_mean = (df_train['boneage'].mean() + df_val['boneage'].mean()) / 2
    boneage_std = (df_train['boneage'].std() + df_val['boneage'].std()) / 2
    
    # Setup paths
    test_path = 'Data/test_CLAHE/'
    img_size = (500, 500)
    
    # Prepare gender data for dual input models
    if args.model_type in ['sex', 'attn_sex']:
        test_gender = df_test['sex'].values
        print(f'Using gender information for {args.model_type} model.')
    else:
        test_gender = None
    
    # Calculate steps
    step_size_test = len(df_test) // args.batch_size_test
    print(f'Test set size: {len(df_test)}')
    print(f'Batch size: {args.batch_size_test}')
    print(f'Step size: {step_size_test}')
    print(f'Expected total predictions: {step_size_test * args.batch_size_test}')
    
    # Create custom objects for model loading
    custom_objects = {}
    if args.loss_type == 'huber':
        custom_objects['loss'] = huber_loss(args.huber_delta)
        print(f'Using Huber loss with delta={args.huber_delta}')
    elif args.loss_type == 'smooth_l1':
        custom_objects['loss'] = smooth_l1_loss(args.smooth_l1_sigma)
        print(f'Using Smooth L1 loss with sigma={args.smooth_l1_sigma}')
    elif args.loss_type == 'wing':
        custom_objects['loss'] = wing_loss(args.wing_w, args.wing_epsilon)
        print(f'Using Wing loss with w={args.wing_w}, epsilon={args.wing_epsilon}')
    else:
        custom_objects['loss'] = 'mae'
        print('Using MAE loss')
    
    # Load the model
    print(f'Loading model from {args.model_path}...')
    try:
        model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)
        print('Model loaded successfully!')
    except Exception as e:
        print(f'Error loading model: {e}')
        return
    
    # Create fresh generator for test evaluation
    print('Creating fresh generator for test evaluation...')
    test_data_eval = create_fresh_test_generator(
        df_test, test_path, test_gender, args.batch_size_test, args.seed, img_size, args.model_type
    )
    
    # Evaluate model
    test_loss, test_mae = model.evaluate(test_data_eval, steps=step_size_test, verbose=1)
    print(f'Test loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}')
    
    # Create fresh generator for test predictions
    print('Creating fresh generator for test predictions...')
    test_data_pred = create_fresh_test_generator(
        df_test, test_path, test_gender, args.batch_size_test, args.seed, img_size, args.model_type
    )
    
    # Generate predictions
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
    
    # Handle shape mismatch - this can happen if some images couldn't be loaded
    print(f'Shape of true bone ages: {true_bone_ages.shape}')
    print(f'Shape of predicted bone ages: {predicted_bone_ages.shape}')
    
    if len(true_bone_ages) != len(predicted_bone_ages):
        print(f'WARNING: Shape mismatch! True: {len(true_bone_ages)}, Predicted: {len(predicted_bone_ages)}')
        # Use the smaller length to avoid broadcasting error
        min_length = min(len(true_bone_ages), len(predicted_bone_ages))
        true_bone_ages = true_bone_ages[:min_length]
        predicted_bone_ages = predicted_bone_ages[:min_length]
        image_ids = df_test['id'].values[:min_length]  # Also truncate image IDs
        print(f'Truncated all arrays to length: {min_length}')
    else:
        image_ids = df_test['id'].values
    
    # Create predictions CSV filename
    model_name = os.path.basename(args.model_path).replace('.h5', '')
    csv_path = f'test_results_{model_name}_predictions.csv'
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'image_id': image_ids,
        'true_bone_age': true_bone_ages,
        'predicted_bone_age': predicted_bone_ages,
        'absolute_error': np.abs(true_bone_ages - predicted_bone_ages)
    })
    predictions_df.to_csv(csv_path, index=False)
    print(f'Predictions saved to {csv_path}')
    
    # Calculate actual performance metrics
    actual_mae = predictions_df['absolute_error'].mean()
    actual_corr, _ = pearsonr(true_bone_ages, predicted_bone_ages)
    
    print(f'\n=== TEST RESULTS ===')
    print(f'Model: {args.model_path}')
    print(f'Model type: {args.model_type}')
    print(f'Loss function: {args.loss_type}')
    print(f'MAE in months: {actual_mae:.2f}')
    print(f'Correlation: {actual_corr:.3f}')
    print(f'Z-score MAE: {test_mae:.4f}')
    
    if args.use_tta:
        print(f'TTA steps: {args.tta_steps}')

if __name__ == '__main__':
    main() 