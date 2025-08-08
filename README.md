# BoneAge Project

This project is for bone age assessment using deep learning on hand X-ray images. It includes scripts for data preprocessing, model training, benchmarking inference speed, and data quality checking.

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
├── clahe/                 # CLAHE utilities (apply to val/test + README)
├── inference_speed/       # Inference speed tools (compute/analyze + README)
├── util/                  # Training utilities (loss functions, augmentation, CLR)
├── check_nan_in_dfs.py    # Data/image quality check and visualization
├── train_baseline.py      # Baseline model training script
├── train_advanced.py      # Advanced/flexible model training script
├── train_advanced_FIXED.py
├── train_advanced_FIXED_enhanced.py
├── modeling.py            # Baseline model architecture utilities
├── modeling_advanced.py   # Advanced model architecture utilities
├── preprocessing.py       # Image/data preprocessing utilities
├── preprocessing_enhanced.py
├── requirements.txt       # Python dependencies
├── README.md              # Project instructions and documentation
├── ADVANCED_ATTENTION_README.md  # Advanced attention mechanisms documentation
├── LICENSE                # License file
├── modeling_notebook.ipynb
└── venv/                  # (Optional) Python virtual environment
```

---

## Setup

1. Clone the repository and navigate to the project directory.
2. Create and activate the virtual environment in `venv/` (recommended), then install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare data in `Data/` with subfolders `train/`, `validation/`, `test/` and CSVs `df_train.csv`, `df_val.csv`, `df_test.csv`.

---

## Check Data Quality and Visualize Images

- **Script:** `check_nan_in_dfs.py`
- **Usage:** `python check_nan_in_dfs.py`
- Checks CSVs for NaN/Inf, flags bad rows/files, and shows sample images (original vs CLAHE) and distributions.

---

## CLAHE Utilities

- Folder: `clahe/`
- Scripts to apply CLAHE to validation and test images only. Run from project root:
  - Validation: `python -m clahe.apply_validation`
  - Test: `python -m clahe.apply_test`
- Output folders:
  - `Data/validation_CLAHE/`
  - `Data/test_CLAHE/`
- Important: Applying CLAHE across all train, validation, and test images resulted in worse performance in our experiments due to a shift in pixel intensity distribution. While CLAHE enhances visualization, it is not consistent and can degrade accuracy when applied broadly. Prefer limiting CLAHE to validation/test or validate carefully.

---

## Train Models

### Baseline
- **Script:** `train_baseline.py`
- **Usage:** `python train_baseline.py`

### Advanced
- **Script:** `train_advanced.py`
- Key options include `--backbone`, `--activation`, `--dropout_rate`, `--dense_units`, `--weights`, `--epochs_frozen`, `--epochs_finetune`, `--fine_tune`.
- Use CLAHE val/test by adding `--use_clahe_val_test`.
- Example:
  ```bash
  python train_advanced.py --backbone xception --epochs_frozen 20 --fine_tune --use_clahe_val_test
  ```

### Enhanced Training (with Utilities)
- **Script:** `train_advanced_FIXED_enhanced.py` (original) or `train_advanced_streamlined.py` (recommended)
- **Features:** Uses the new `util/` package with enhanced loss functions, cyclical learning rates, data augmentation, and test-time augmentation.
- **Key enhancements:**
  - Cyclical Learning Rate: `--use_cyclical_lr`
  - Enhanced loss functions: `--loss_type` (mae, mse, huber, smooth_l1, wing)
  - Label noise: `--label_noise_std`
  - Mixup augmentation: `--mixup_alpha`
  - Test Time Augmentation: `--use_tta`
- **Example:**
  ```bash
  # Using the streamlined version (recommended)
  python train_advanced_streamlined.py --backbone xception --use_cyclical_lr --loss_type huber --label_noise_std 0.01 --use_tta
  
  # Using the original enhanced version
  python train_advanced_FIXED_enhanced.py --backbone xception --use_cyclical_lr --loss_type huber --label_noise_std 0.01 --use_tta
  ```

---

## Run All Backbones Experiment

- **Script:** `run_all_backbones.py`
- Automates running `train_advanced.py` across backbones and saves JSON results in `backbone_results/`.

---

## Training Utilities

- **Folder:** `util/`
- **Features:** Enhanced loss functions, cyclical learning rates, data augmentation, and test-time augmentation
- **Modules:**
  - `cyclical_lr.py`: Cyclical Learning Rate callback
  - `loss_functions.py`: Enhanced loss functions (Huber, Smooth L1, Wing)
  - `augmentation.py`: Data augmentation (mixup, label noise, enhanced generators)
  - `tta.py`: Test Time Augmentation
- **Usage:** Import utilities in training scripts or use the enhanced training script
- **See:** `util/README.md` for detailed documentation and examples

## Attention Models

### Original Attention Model
- **File:** `modeling_advanced.py` - `attn_sex_model()` function
- **Features:** Original attention mechanism with gender incorporation
- **Usage:** Standard attention model with simple concatenation of image and clinical features

### Improved Attention Model
- **File:** `modeling_advanced.py` - `attn_sex_model_improved()` function
- **Features:** Focused attention mechanisms for bone age prediction
- **Key improvements:**
  - Cross-Attention between image and clinical (sex) features for better feature interaction
  - Gated Feature Fusion for adaptive feature combination with interpretable weights
- **Research basis:** Incorporates proven techniques from recent computer vision and medical imaging literature
- **Usage:** Compatible with all training scripts, maintains same interface as original
- **Performance:** Expected 3-8% MAE improvement over baseline attention model
- **Benefits:** Focused approach prevents overfitting while providing significant performance gains
- **See:** `ADVANCED_ATTENTION_README.md` for detailed technical documentation and research references

## Inference Speed Tools

- Folder: `inference_speed/`
- Compute: `python -m inference_speed.compute_inference_speed`
- Analyze: `python -m inference_speed.analyze_inference_results`
- See `inference_speed/README.md` for usage details.

---

## Tips
- Control Keras `fit()` verbosity via the `verbose` parameter.
- Ensure `modeling_advanced.py` remains importable from project root when running package modules with `-m`.