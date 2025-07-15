# Inference Speed Measurement Tools

This directory contains tools to measure and analyze inference speed for different backbone architectures in your bone age prediction models.

## Files Created

1. **`compute_inference_speed.py`** - Main script to measure inference speeds
2. **`analyze_inference_results.py`** - Script to analyze and visualize results
3. **`example_run_inference_speed.py`** - Example script showing different usage scenarios

## Quick Start

### 1. Basic Usage
Run inference speed measurement for all available backbones:
```bash
python compute_inference_speed.py
```

### 2. Test Specific Backbones
```bash
python compute_inference_speed.py --backbones xception resnet50 mobilenet_v3_small
```

### 3. Custom Batch Sizes
```bash
python compute_inference_speed.py --batch_sizes 1 4 8 16 32 64
```

### 4. Quick Test (Fewer Runs)
```bash
python compute_inference_speed.py --num_runs 5 --batch_sizes 1 8
```

## Advanced Options

### Complete Parameter List
```bash
python compute_inference_speed.py \
    --img_height 500 \
    --img_width 500 \
    --img_channels 3 \
    --batch_sizes 1 4 8 16 32 \
    --num_runs 10 \
    --backbones xception resnet50 mobilenet_v3_small
```

### Available Backbones
- `xception`
- `resnet50`, `resnet101`, `resnet152`
- `resnet50v2`, `resnet101v2`, `resnet152v2`
- `mobilenet`, `mobilenet_v2`
- `mobilenet_v3_small`, `mobilenet_v3_large`
- `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`

## Results Analysis

### Analyze Latest Results
```bash
python analyze_inference_results.py
```

### Analyze Specific CSV File
```bash
python analyze_inference_results.py --csv_file inference_results/inference_speed_20240108_143022.csv
```

### Save Plots to File
```bash
python analyze_inference_results.py --save_plots
```

### Skip Plots (Text Summary Only)
```bash
python analyze_inference_results.py --no_plots
```

## Output Files

Results are saved in the `inference_results/` directory:

- **`inference_speed_YYYYMMDD_HHMMSS.csv`** - Detailed measurements
- **`speed_comparison.csv`** - Simplified comparison table
- **`inference_analysis_plots.png`** - Visualization plots (if --save_plots used)

## CSV Output Columns

### Detailed Results (`inference_speed_*.csv`)
- `backbone` - Architecture name
- `batch_size` - Number of images processed together
- `mean_inference_time_s` - Average time for the batch
- `time_per_image_s` - Average time per single image
- `time_per_image_ms` - Average time per image in milliseconds
- `images_per_second` - Throughput (images processed per second)
- `total_params` - Total model parameters
- `std_inference_time_s` - Standard deviation of timing measurements
- `min_inference_time_s`, `max_inference_time_s` - Min/max times observed
- System info (GPU, TensorFlow version, etc.)

### Comparison Results (`speed_comparison.csv`)
- `backbone` - Architecture name
- `single_image_time_ms` - Time to process one image
- `single_image_throughput` - Images per second (batch size 1)
- `best_throughput` - Highest throughput achieved
- `best_throughput_batch_size` - Batch size that achieved best throughput
- `parameters_millions` - Model size in millions of parameters
- `efficiency_img_per_sec_per_million_params` - Speed/size efficiency metric

## Example Output

```
=== Summary (Images per second at batch size 1) ===
mobilenet_v3_small  :   45.2 img/s
mobilenet           :   42.1 img/s
mobilenet_v2        :   38.7 img/s
xception            :   28.5 img/s
resnet50            :   25.3 img/s
convnext_tiny       :   22.1 img/s
```

## Tips for Usage

1. **GPU vs CPU**: Results will vary significantly between GPU and CPU. The script automatically detects and uses available GPUs.

2. **Batch Size Impact**: Larger batch sizes generally increase total throughput but also increase latency per batch.

3. **Multiple Runs**: The script runs multiple iterations and reports statistics to account for timing variation.

4. **Memory Considerations**: Large models with large batch sizes may exceed GPU memory. The script handles this gracefully.

5. **Fair Comparison**: All models use the same input size (500x500x3 by default) and similar configurations for fair comparison.

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch sizes or test fewer models at once
2. **Import errors**: Ensure all dependencies are installed (`pip install -r requirements.txt`)
3. **No results**: Check that your modeling_advanced.py module is importable

### Dependencies
- tensorflow
- pandas
- numpy
- matplotlib (for analysis plots)
- seaborn (for analysis plots)
- psutil (for system info)

## Example Workflow

```bash
# 1. Quick test with a few models
python compute_inference_speed.py --backbones xception mobilenet_v3_small --num_runs 5

# 2. Full benchmark (takes longer)
python compute_inference_speed.py

# 3. Analyze results
python analyze_inference_results.py

# 4. Get just the summary without plots
python analyze_inference_results.py --no_plots
```

This will give you comprehensive insights into which backbone architectures are fastest for your specific hardware setup and use case. 