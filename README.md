# BoneAge Project

This project is for bone age assessment using deep learning on hand X-ray images. It includes scripts for data preprocessing, model training, and data quality checking.

---

## Folder Structure Reference

```
BoneAge/
├── Data/                  # All data files and subfolders (images, CSVs)
│   ├── train/             # Training images
│   ├── validation/        # Validation images
│   ├── test/              # Test images
│   ├── validation_CLAHE/  # CLAHE-enhanced validation images (created by script)
│   ├── test_CLAHE/        # CLAHE-enhanced test images (created by script)
│   ├── df_train.csv       # Training set metadata
│   ├── df_val.csv         # Validation set metadata
│   ├── df_test.csv        # Test set metadata
│   └── ...                # Other CSVs or data splits
├── CLAHE_apply_validation.py # Script to apply CLAHE to validation images
├── CLAHE_apply_test.py       # Script to apply CLAHE to test images
├── check_nan_in_dfs.py       # Data/image quality check and visualization
├── train_baseline.py         # Baseline model training script
├── train_advanced.py         # Advanced/flexible model training script
├── modeling.py               # Baseline model architecture utilities
├── modeling_advanced.py      # Advanced model architecture utilities
├── preprocessing.py          # Image/data preprocessing utilities
├── requirements.txt          # Python dependencies
├── README.md                 # Project instructions and documentation
├── LICENSE                   # License file
├── best_model.h5             # (Optional) Saved best model weights
├── baseline_model.h5         # (Optional) Saved baseline model weights
├── baseline_training_log.csv # (Optional) Training log
├── modeling_notebook.ipynb   # (Optional) Jupyter notebook for exploration
├── .gitignore                # Git ignore rules
└── venv/                     # (Optional) Python virtual environment
```

---

## Step-by-Step Instructions

### 1. **Setup**

1. **Clone the repository** and navigate to the project directory.
2. **Install dependencies** (preferably in a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare data**: Place your data in the `Data/` directory, with subfolders for `train/`, `validation/`, and `test/` images, and the corresponding CSV files (e.g., `df_train.csv`, `df_val.csv`, `df_test.csv`).

---

### 2. **Check Data Quality and Visualize Images**

**Script:** `check_nan_in_dfs.py`

- **Purpose:**  
  This script checks your dataframes and images for issues that can cause model training to fail (e.g., NaN/Inf values, missing or corrupt images).  
  It also displays sample images from the train, validation, and test sets, both in their original form and with CLAHE enhancement, so you can visually inspect the effect of CLAHE and the quality of your data.

- **What it does:**
  - Reports NaN and Inf values in `df_train.csv`, `df_val.csv`, and `df_test.csv`.
  - Reports rows with missing or problematic values.
  - Samples 20 images from each of train, validation, and test sets:
    - Reports file type, pixel type, average file size, and average dimensions.
    - Plots pixel value distributions for each set.
  - **Displays side-by-side images:** For each set, shows a sample image in both original and CLAHE-enhanced form.

- **How to use:**
  ```bash
  python check_nan_in_dfs.py
  ```
  - Review the output for any NaN/Inf rows or image anomalies.
  - Visually inspect the displayed images to see the effect of CLAHE and check for any image quality issues.
  - Remove or fix problematic rows/files before training.

---

### 3. **Create CLAHE-Enhanced Images**

**Scripts:**  
- `CLAHE_apply_validation.py`  
- `CLAHE_apply_test.py`  

- **Purpose:**  
  These scripts apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to all validation and test images, creating new datasets with enhanced contrast. This can improve model performance, especially on lower-quality images.

- **How to use:**
  ```bash
  python CLAHE_apply_validation.py
  python CLAHE_apply_test.py
  ```
  - The enhanced images will be saved in the respective output folders with the same filenames as the originals:
    - `Data/validation_CLAHE/`
    - `Data/test_CLAHE/`

---

### 4. **Train a Model**

#### **A. Baseline Model**

**Script:** `train_baseline.py`

- **Purpose:**  
  Trains a simple baseline deep learning model for bone age prediction using a fixed architecture (Xception backbone, fixed hyperparameters).

- **How to use:**
  ```bash
  python train_baseline.py
  ```

---

#### **B. Advanced Model**

**Script:** `train_advanced.py`

- **Purpose:**  
  Trains a more flexible and customizable model. You can select different backbones, activation functions, dropout rates, dense layer sizes, and more.  
  You can also choose to use the CLAHE-enhanced validation and test images by adding the `--use_clahe_val_test` flag.

- **How to use:**
  ```bash
  # To use original validation/test images:
  python train_advanced.py

  # To use CLAHE-enhanced validation/test images:
  python train_advanced.py --use_clahe_val_test
  ```

- **Key options:**  
  - `--backbone`: Choose model backbone (e.g., xception, resnet50, mobilenet, efficientnetb0, etc.)
  - `--activation`: Activation function for dense layer (e.g., relu, tanh, swish)
  - `--dropout_rate`: Dropout rate after pooling
  - `--dense_units`: Number of units in dense layer
  - `--weights`: Pretrained weights to use (imagenet or None)
  - `--epochs_frozen`: Epochs to train with frozen base
  - `--epochs_finetune`: Epochs to train with unfrozen base
  - `--fine_tune`: Enable fine-tuning after initial training
  - `--use_clahe_val_test`: Use CLAHE-enhanced validation and test images

---

## **How are the advanced scripts different from the originals?**

### **train_advanced.py vs. train_baseline.py**
- `train_baseline.py` uses a fixed model architecture (Xception backbone, fixed hyperparameters) and is intended as a simple starting point.
- `train_advanced.py` allows you to:
  - Select from multiple backbone architectures (Xception, ResNet, MobileNet, EfficientNet, etc.).
  - Customize activation functions, dropout rates, dense layer sizes, and more.
  - Fine-tune the model by unfreezing layers after initial training.
  - Use CLAHE-enhanced validation and test images with a simple flag.
  - Pass all options via command-line arguments for reproducibility and experimentation.

### **modeling_advanced.py vs. modeling.py**
- `modeling.py` contains basic model-building utilities for the baseline model.
- `modeling_advanced.py` provides:
  - Support for multiple backbones and more flexible model architectures.
  - Improved weight initialization strategies based on activation function.
  - Utility functions for fine-tuning (unfreezing layers), plotting training history, and more.
  - More modular and extensible code for advanced experimentation.

---

## **Output Folders**
- `Data/validation_CLAHE/`: CLAHE-enhanced validation images
- `Data/test_CLAHE/`: CLAHE-enhanced test images

---

## **Troubleshooting**
- If you see NaN loss/MAE during training, check your data for NaN/Inf values using `check_nan_in_dfs.py`.
- Ensure all image files referenced in the CSVs exist and are readable.
- Check that all dependencies are installed (see `requirements.txt`).

---

## **License**
See `LICENSE` for details.