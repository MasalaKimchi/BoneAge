# Inference Speed Measurement Tools

This folder contains tools to measure and analyze inference speed for different backbone architectures in the BoneAge project.

## Quick Start

Run these from the project root so imports resolve correctly (or use your virtual environment in `venv`).

### 1. Basic Usage
```bash
python -m inference_speed.compute_inference_speed
```

### 2. Test Specific Backbones
```bash
python -m inference_speed.compute_inference_speed --backbones xception resnet50 mobilenet_v3_small
```

### 3. Custom Batch Sizes
```bash
python -m inference_speed.compute_inference_speed --batch_sizes 1 4 8 16 32 64
```

### 4. Quick Test (Fewer Runs)
```bash
python -m inference_speed.compute_inference_speed --num_runs 5 --batch_sizes 1 8
```

## Analysis

### Analyze Latest Results
```bash
python -m inference_speed.analyze_inference_results
```

### Analyze Specific CSV File
```bash
python -m inference_speed.analyze_inference_results --csv_file inference_results/inference_speed_YYYYMMDD_HHMMSS.csv
```

### Save Plots
```bash
python -m inference_speed.analyze_inference_results --save_plots
```

### Skip Plots (Text Summary Only)
```bash
python -m inference_speed.analyze_inference_results --no_plots
```

## Output Files

Results are saved in the `inference_results/` directory:
- `inference_speed_YYYYMMDD_HHMMSS.csv` – detailed measurements
- `speed_comparison.csv` – simplified comparison
- `inference_analysis_plots.png` – plots (when `--save_plots` is used)

## Backbones
Supported backbones come from `modeling_advanced.py` and include:
- xception
- resnet50/resnet101/resnet152 (+ v2 variants)
- mobilenet, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
- convnext_tiny/small/base/large

## Tips
- Larger batches increase throughput but may raise latency and memory use.
- GPU vs CPU performance differs greatly; the script reports system info.
- If you hit OOM, reduce batch sizes or test fewer models.

## Dependencies
- tensorflow, pandas, numpy, psutil
- matplotlib, seaborn (analysis only)

## Example Workflow
```bash
# 1. Quick test
python -m inference_speed.compute_inference_speed --backbones xception mobilenet_v3_small --num_runs 5

# 2. Full benchmark
python -m inference_speed.compute_inference_speed

# 3. Analyze results
python -m inference_speed.analyze_inference_results
``` 