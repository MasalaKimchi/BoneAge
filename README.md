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

---

## Run All Backbones Experiment

- **Script:** `run_all_backbones.py`
- Automates running `train_advanced.py` across backbones and saves JSON results in `backbone_results/`.

---

## Inference Speed Tools

- Folder: `inference_speed/`
- Compute: `python -m inference_speed.compute_inference_speed`
- Analyze: `python -m inference_speed.analyze_inference_results`
- See `inference_speed/README.md` for usage details.

---

## Tips
- Control Keras `fit()` verbosity via the `verbose` parameter.
- Ensure `modeling_advanced.py` remains importable from project root when running package modules with `-m`.