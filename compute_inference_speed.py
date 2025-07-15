"""
Inference Speed Measurement Script

This script measures inference speed for different backbone architectures
and saves the results to a CSV file. It tests various batch sizes and 
provides comprehensive timing metrics.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os
import argparse
from datetime import datetime
import gc
import psutil
import subprocess

# Import local modules
import modeling_advanced as mod


def get_gpu_info():
    """Get GPU information for context."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu_details = []
            for i, gpu in enumerate(gpus):
                gpu_name = tf.config.experimental.get_device_details(gpu).get('device_name', 'Unknown')
                gpu_details.append(f"GPU {i}: {gpu_name}")
            return ", ".join(gpu_details)
        else:
            return "No GPU detected"
    except:
        return "GPU info unavailable"


def get_system_info():
    """Get system information for context."""
    try:
        cpu_info = f"{psutil.cpu_count()} cores"
        memory_info = f"{psutil.virtual_memory().total // (1024**3)} GB RAM"
        return f"{cpu_info}, {memory_info}"
    except:
        return "System info unavailable"


def create_dummy_images(batch_size, img_dims=(500, 500, 3)):
    """Create dummy images for inference timing."""
    return tf.random.normal((batch_size,) + img_dims, dtype=tf.float32)


def create_model_for_backbone(backbone_name, img_dims=(500, 500, 3)):
    """Create a model for the specified backbone."""
    try:
        # Use a minimal optimizer for speed testing
        optim = tf.keras.optimizers.Adam(learning_rate=1e-4)
        
        model = mod.baseline_model(
            img_dims=img_dims,
            activation='relu',  # Fast activation
            optim=optim,
            metric=['mae'],
            backbone=backbone_name,
            weights='imagenet',
            dropout_rate=0.0,  # Disable dropout for consistent timing
            dense_units=500
        )
        
        # Compile model
        model.compile(optimizer=optim, loss='mae', metrics=['mae'])
        
        return model
    except Exception as e:
        print(f"Error creating model for backbone {backbone_name}: {e}")
        return None


def warmup_model(model, batch_size, img_dims, warmup_runs=3):
    """Warm up the model with a few inference runs."""
    print(f"  Warming up model with batch size {batch_size}...")
    for _ in range(warmup_runs):
        dummy_batch = create_dummy_images(batch_size, img_dims)
        _ = model.predict(dummy_batch, verbose=0)


def measure_inference_time(model, batch_size, img_dims, num_runs=10):
    """Measure inference time for a given batch size."""
    times = []
    
    for run in range(num_runs):
        # Create fresh batch for each run
        dummy_batch = create_dummy_images(batch_size, img_dims)
        
        # Time the inference
        start_time = time.perf_counter()
        predictions = model.predict(dummy_batch, verbose=0)
        end_time = time.perf_counter()
        
        inference_time = end_time - start_time
        times.append(inference_time)
        
        # Clean up
        del dummy_batch, predictions
        
    return times


def benchmark_backbone(backbone_name, img_dims=(500, 500, 3), batch_sizes=[1, 4, 8, 16, 32], num_runs=10):
    """Benchmark a single backbone architecture."""
    print(f"\nBenchmarking {backbone_name}...")
    
    # Create model
    model = create_model_for_backbone(backbone_name, img_dims)
    if model is None:
        return []
    
    results = []
    
    try:
        # Get model size information
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            try:
                # Warmup
                warmup_model(model, batch_size, img_dims, warmup_runs=3)
                
                # Measure inference times
                times = measure_inference_time(model, batch_size, img_dims, num_runs)
                
                # Calculate statistics
                mean_time = np.mean(times)
                std_time = np.std(times)
                min_time = np.min(times)
                max_time = np.max(times)
                median_time = np.median(times)
                
                # Calculate derived metrics
                time_per_image = mean_time / batch_size
                images_per_second = batch_size / mean_time
                
                # Store results
                result = {
                    'backbone': backbone_name,
                    'batch_size': batch_size,
                    'mean_inference_time_s': mean_time,
                    'std_inference_time_s': std_time,
                    'min_inference_time_s': min_time,
                    'max_inference_time_s': max_time,
                    'median_inference_time_s': median_time,
                    'time_per_image_s': time_per_image,
                    'images_per_second': images_per_second,
                    'time_per_image_ms': time_per_image * 1000,
                    'total_params': total_params,
                    'trainable_params': trainable_params,
                    'non_trainable_params': non_trainable_params,
                    'num_runs': num_runs,
                    'img_height': img_dims[0],
                    'img_width': img_dims[1],
                    'img_channels': img_dims[2]
                }
                results.append(result)
                
                print(f"    Mean time: {mean_time:.4f}s, Time per image: {time_per_image*1000:.2f}ms, Throughput: {images_per_second:.1f} img/s")
                
            except Exception as e:
                print(f"    Error with batch size {batch_size}: {e}")
                continue
        
    except Exception as e:
        print(f"  Error benchmarking {backbone_name}: {e}")
    
    finally:
        # Clean up model
        del model
        tf.keras.backend.clear_session()
        gc.collect()
    
    return results


def run_inference_benchmark(args):
    """Run the complete inference benchmark."""
    print("=== Inference Speed Benchmark ===")
    print(f"Image dimensions: {args.img_height}x{args.img_width}x{args.img_channels}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Number of runs per test: {args.num_runs}")
    print(f"GPU Info: {get_gpu_info()}")
    print(f"System Info: {get_system_info()}")
    
    # Get available backbones
    if args.backbones:
        selected_backbones = args.backbones
    else:
        selected_backbones = list(mod.BACKBONE_MAP.keys())
    
    print(f"Testing backbones: {selected_backbones}")
    
    img_dims = (args.img_height, args.img_width, args.img_channels)
    all_results = []
    
    # Benchmark each backbone
    for backbone_name in selected_backbones:
        if backbone_name in mod.BACKBONE_MAP:
            results = benchmark_backbone(
                backbone_name, 
                img_dims=img_dims, 
                batch_sizes=args.batch_sizes, 
                num_runs=args.num_runs
            )
            all_results.extend(results)
        else:
            print(f"Warning: Backbone '{backbone_name}' not found in BACKBONE_MAP")
    
    # Create DataFrame and save results
    if not all_results:
        print("No results obtained!")
        return None
        
    df = pd.DataFrame(all_results)
    
    # Add metadata
    df['timestamp'] = datetime.now().isoformat()
    df['tensorflow_version'] = tf.__version__
    df['gpu_info'] = get_gpu_info()
    df['system_info'] = get_system_info()
    
    # Sort by backbone and batch size
    df = df.sort_values(['backbone', 'batch_size'])
    
    # Save to CSV
    os.makedirs('inference_results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'inference_results/inference_speed_{timestamp}.csv'
    df.to_csv(csv_filename, index=False)
    
    print(f"\n=== Results saved to {csv_filename} ===")
    
    # Display summary
    print("\n=== Summary (Mean time per image in ms) ===")
    summary = df.pivot(index='backbone', columns='batch_size', values='time_per_image_ms')
    print(summary.round(2))
    
    print("\n=== Summary (Images per second at batch size 1) ===")
    batch1_data = df[df['batch_size'] == 1][['backbone', 'images_per_second']]
    batch1_summary = batch1_data.sort_values('images_per_second', ascending=False)
    for _, row in batch1_summary.iterrows():
        print(f"{row['backbone']:20}: {row['images_per_second']:6.1f} img/s")
        
    return csv_filename


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Measure inference speed for different backbone architectures')
    
    parser.add_argument('--img_height', type=int, default=500, help='Image height (default: 500)')
    parser.add_argument('--img_width', type=int, default=500, help='Image width (default: 500)')
    parser.add_argument('--img_channels', type=int, default=3, help='Image channels (default: 3)')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 4, 8, 16, 32], 
                       help='Batch sizes to test (default: 1 4 8 16 32)')
    parser.add_argument('--num_runs', type=int, default=10, 
                       help='Number of timing runs per test (default: 10)')
    parser.add_argument('--backbones', type=str, nargs='+', 
                       help='Specific backbones to test (default: all available)')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Set memory growth for GPU if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory growth setting failed: {e}")
    
    # Run benchmark
    csv_filename = run_inference_benchmark(args)
    
    if csv_filename:
        print(f"\n✅ Inference speed measurement completed!")
        print(f"📄 Results saved to: {csv_filename}")
        print(f"\n💡 To run specific backbones: python compute_inference_speed.py --backbones xception resnet50")
        print(f"💡 To test different batch sizes: python compute_inference_speed.py --batch_sizes 1 8 16")
    else:
        print("❌ Inference speed measurement failed!") 