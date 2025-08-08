# CLAHE Utilities

Scripts to apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to validation and test images.

## Important
- Applying CLAHE across all train, validation, and test images resulted in worse performance in our experiments due to a shift in pixel intensity distribution. While CLAHE can enhance visualization, it introduces inconsistency across datasets. Prefer applying CLAHE only to validation and test, or run targeted experiments before using it broadly.

## Usage (run from project root)

### Apply to validation images
```bash
python -m clahe.apply_validation
```

### Apply to test images
```bash
python -m clahe.apply_test
```

Outputs are written to:
- `Data/validation_CLAHE/`
- `Data/test_CLAHE/`

Input CSVs and image folders are read from:
- `Data/df_val.csv` and `Data/validation/`
- `Data/df_test.csv` and `Data/test/` 