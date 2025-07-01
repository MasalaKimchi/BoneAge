import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# Paths to CSVs
train_csv = './Data/df_train.csv'
val_csv = './Data/df_val.csv'
test_csv = './Data/df_test.csv'

# Image directories
train_img_dir = './Data/train/'
val_img_dir = './Data/validation/'
test_img_dir = './Data/test/'

# Load dataframes
df_train = pd.read_csv(train_csv)
df_val = pd.read_csv(val_csv)
df_test = pd.read_csv(test_csv)

def check_df(df, name):
    print(f'\nChecking {name}...')
    print(df.info())
    print(df.describe(include="all"))
    print('NaN count per column:')
    print(df.isna().sum())
    print('Inf count per column:')
    print(np.isinf(df.select_dtypes(include=[np.number])).sum())
    # Show rows with any NaN or Inf
    nan_rows = df[df.isna().any(axis=1)]
    inf_rows = df[np.isinf(df.select_dtypes(include=[np.number])).any(axis=1)]
    if not nan_rows.empty:
        print('Rows with NaN values:')
        print(nan_rows)
    if not inf_rows.empty:
        print('Rows with Inf values:')
        print(inf_rows)
    print('---')

def sample_image_stats(df, img_dir, name, n=20):
    print(f'\nImage stats for {name}:')
    sample = df.dropna(subset=['filename']).sample(n=min(n, len(df)), random_state=42)
    file_types = []
    pixel_types = []
    sizes = []
    dims = []
    pixel_values = []
    for fname in sample['filename']:
        path = os.path.join(img_dir, fname)
        if not os.path.exists(path):
            print(f'File not found: {path}')
            continue
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'Could not read: {path}')
            continue
        file_types.append(os.path.splitext(fname)[1])
        pixel_types.append(img.dtype)
        sizes.append(os.path.getsize(path))
        dims.append(img.shape)
        pixel_values.append(img.flatten())
    if file_types:
        print('File types:', set(file_types))
        print('Pixel types:', set(pixel_types))
        print('Avg file size (KB):', np.mean(sizes)/1024)
        print('Avg dimensions:', np.mean([d[0] for d in dims]), 'x', np.mean([d[1] for d in dims]))
    else:
        print('No images found/readable.')
    return pixel_values

# Run checks
def main():
    check_df(df_train, 'df_train.csv')
    check_df(df_val, 'df_val.csv')
    check_df(df_test, 'df_test.csv')

    # Image stats and pixel distributions
    train_pixels = sample_image_stats(df_train, train_img_dir, 'train')
    val_pixels = sample_image_stats(df_val, val_img_dir, 'validation')
    test_pixels = sample_image_stats(df_test, test_img_dir, 'test')

    # Plot pixel value distributions
    plt.figure(figsize=(15, 5))
    for i, (pixels, title) in enumerate(zip([train_pixels, val_pixels, test_pixels], ['Train', 'Validation', 'Test'])):
        plt.subplot(1, 3, i+1)
        if pixels:
            all_pixels = np.concatenate(pixels)
            plt.hist(all_pixels, bins=50, color='blue', alpha=0.7)
            plt.title(f'{title} Pixel Value Distribution')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
        else:
            plt.title(f'{title} (no images)')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main() 