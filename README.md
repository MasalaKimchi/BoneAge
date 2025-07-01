# BoneAge Project

This project is for bone age assessment using deep learning on hand X-ray images. It includes scripts for data preprocessing, model training, and data quality checking.

## Setup

1. **Clone the repository** and navigate to the project directory.
2. **Install dependencies** (preferably in a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare data**: Place your data in the `Data/` directory, with subfolders for `train/`, `validation/`, and `test/` images, and the corresponding CSV files (e.g., `df_train.csv`, `df_val.csv`, `df_test.csv`).

## Data Quality Check: `check_nan_in_dfs.py`

This script checks your dataframes and images for issues that can cause model training to fail (e.g., NaN/Inf values, missing or corrupt images).

**What it does:**
- Reports NaN and Inf values in `df_train.csv`, `df_val.csv`, and `df_test.csv`.
- Reports rows with missing or problematic values.
- Samples 20 images from each of train, validation, and test sets:
  - Reports file type, pixel type, average file size, and average dimensions.
  - Plots pixel value distributions for each set.

**How to use:**
```bash
python check_nan_in_dfs.py
```
- Review the output for any NaN/Inf rows or image anomalies.
- Remove or fix problematic rows/files before training.

## Model Training: `train_baseline.py`

This script trains a baseline deep learning model for bone age prediction.

**How to use:**
```bash
python train_baseline.py
```

**Replicating the NaN validation loss/MAE error:**
- If your data contains NaN values in the validation or test CSVs (e.g., missing `boneage` or `boneage_zscore`), running this script will likely result in NaN loss/MAE during training or validation.
- Use `check_nan_in_dfs.py` to identify and fix these issues before training.

## Troubleshooting
- If you see NaN loss/MAE during training, check your data for NaN/Inf values using `check_nan_in_dfs.py`.
- Ensure all image files referenced in the CSVs exist and are readable.
- Check that all dependencies are installed (see `requirements.txt`).

## Files to Ignore
- Large model files and logs are ignored by git (see `.gitignore`).

## License
See `LICENSE` for details.