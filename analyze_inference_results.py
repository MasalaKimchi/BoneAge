"""
Analysis script for inference speed results.

This script loads CSV files from inference speed measurements and creates
visualizations and summaries to help compare different backbone architectures.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse
from pathlib import Path

def load_latest_results(results_dir='inference_results'):
    """Load the most recent inference speed results."""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory '{results_dir}' not found!")
        return None
    
    csv_files = list(results_path.glob('inference_speed_*.csv'))
    if not csv_files:
        print(f"No inference speed CSV files found in '{results_dir}'!")
        return None
    
    # Get the most recent file
    latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
    print(f"Loading results from: {latest_file}")
    
    try:
        df = pd.read_csv(latest_file)
        return df, latest_file
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def create_summary_table(df):
    """Create a summary table of inference speeds."""
    print("\n" + "="*80)
    print("INFERENCE SPEED SUMMARY")
    print("="*80)
    
    # Summary at batch size 1
    batch1 = df[df['batch_size'] == 1].copy()
    if not batch1.empty:
        batch1_sorted = batch1.sort_values('images_per_second', ascending=False)
        print(f"\n📊 Single Image Inference Speed (Batch Size = 1)")
        print("-" * 60)
        print(f"{'Backbone':<20} {'Time/Image (ms)':<15} {'Images/sec':<12} {'Parameters':<12}")
        print("-" * 60)
        
        for _, row in batch1_sorted.iterrows():
            params_m = row['total_params'] / 1e6
            print(f"{row['backbone']:<20} {row['time_per_image_ms']:>12.1f} {row['images_per_second']:>10.1f} {params_m:>9.1f}M")
    
    # Best throughput comparison
    print(f"\n🚀 Highest Throughput Models")
    print("-" * 40)
    best_throughput = df.loc[df.groupby('backbone')['images_per_second'].idxmax()]
    best_sorted = best_throughput.sort_values('images_per_second', ascending=False)
    
    for i, (_, row) in enumerate(best_sorted.head(5).iterrows()):
        print(f"{i+1}. {row['backbone']}: {row['images_per_second']:.1f} img/s (batch={row['batch_size']})")
    
    # Efficiency (speed vs parameters)
    print(f"\n⚡ Most Efficient Models (Images/sec per Million Parameters)")
    print("-" * 60)
    batch1['efficiency'] = batch1['images_per_second'] / (batch1['total_params'] / 1e6)
    efficient_sorted = batch1.sort_values('efficiency', ascending=False)
    
    for i, (_, row) in enumerate(efficient_sorted.head(5).iterrows()):
        print(f"{i+1}. {row['backbone']}: {row['efficiency']:.2f} img/s/M params")

def plot_inference_speeds(df, save_plots=False):
    """Create visualization plots for inference speeds."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Inference Speed Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Time per image at batch size 1
    batch1 = df[df['batch_size'] == 1].sort_values('time_per_image_ms')
    axes[0, 0].barh(range(len(batch1)), batch1['time_per_image_ms'])
    axes[0, 0].set_yticks(range(len(batch1)))
    axes[0, 0].set_yticklabels(batch1['backbone'], fontsize=8)
    axes[0, 0].set_xlabel('Time per Image (ms)')
    axes[0, 0].set_title('Inference Time per Image (Batch Size = 1)')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Plot 2: Throughput vs Batch Size
    popular_backbones = ['xception', 'resnet50', 'mobilenet_v3_small', 'convnext_tiny']
    available_backbones = [b for b in popular_backbones if b in df['backbone'].values]
    
    for backbone in available_backbones[:4]:  # Limit to 4 for readability
        backbone_data = df[df['backbone'] == backbone]
        axes[0, 1].plot(backbone_data['batch_size'], backbone_data['images_per_second'], 
                       marker='o', label=backbone)
    
    axes[0, 1].set_xlabel('Batch Size')
    axes[0, 1].set_ylabel('Images per Second')
    axes[0, 1].set_title('Throughput vs Batch Size')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: Model Parameters vs Speed
    batch1 = df[df['batch_size'] == 1]
    scatter = axes[1, 0].scatter(batch1['total_params'] / 1e6, batch1['images_per_second'], 
                                alpha=0.7, s=60)
    
    # Add labels for some points
    for i, row in batch1.iterrows():
        if row['backbone'] in ['xception', 'mobilenet', 'resnet50', 'convnext_large']:
            axes[1, 0].annotate(row['backbone'], 
                               (row['total_params'] / 1e6, row['images_per_second']),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    axes[1, 0].set_xlabel('Model Parameters (Millions)')
    axes[1, 0].set_ylabel('Images per Second')
    axes[1, 0].set_title('Model Size vs Inference Speed')
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Batch processing efficiency
    backbones_subset = df['backbone'].unique()[:8]  # Limit for readability
    batch_data = df[df['backbone'].isin(backbones_subset)]
    
    # Create efficiency metric (relative to batch size 1)
    efficiency_data = []
    for backbone in backbones_subset:
        backbone_df = batch_data[batch_data['backbone'] == backbone].sort_values('batch_size')
        if len(backbone_df) > 1:
            batch1_speed = backbone_df[backbone_df['batch_size'] == 1]['images_per_second'].iloc[0]
            for _, row in backbone_df.iterrows():
                if row['batch_size'] > 1:
                    efficiency = row['images_per_second'] / batch1_speed
                    efficiency_data.append({
                        'backbone': backbone,
                        'batch_size': row['batch_size'],
                        'efficiency_ratio': efficiency
                    })
    
    if efficiency_data:
        eff_df = pd.DataFrame(efficiency_data)
        pivot_eff = eff_df.pivot(index='backbone', columns='batch_size', values='efficiency_ratio')
        sns.heatmap(pivot_eff, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1, 1])
        axes[1, 1].set_title('Batch Processing Efficiency\n(Relative to Batch Size 1)')
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor efficiency analysis', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Batch Processing Efficiency')
    
    plt.tight_layout()
    
    if save_plots:
        plot_filename = 'inference_results/inference_analysis_plots.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Plots saved to: {plot_filename}")
    
    plt.show()

def create_comparison_csv(df, output_file='inference_results/speed_comparison.csv'):
    """Create a simplified comparison CSV for easy sharing."""
    
    # Get key metrics for each backbone
    comparison_data = []
    
    for backbone in df['backbone'].unique():
        backbone_data = df[df['backbone'] == backbone]
        
        # Get batch size 1 data
        batch1 = backbone_data[backbone_data['batch_size'] == 1]
        if batch1.empty:
            continue
            
        batch1_row = batch1.iloc[0]
        
        # Get best throughput
        best_throughput_row = backbone_data.loc[backbone_data['images_per_second'].idxmax()]
        
        comparison_data.append({
            'backbone': backbone,
            'single_image_time_ms': batch1_row['time_per_image_ms'],
            'single_image_throughput': batch1_row['images_per_second'],
            'best_throughput': best_throughput_row['images_per_second'],
            'best_throughput_batch_size': best_throughput_row['batch_size'],
            'total_parameters': batch1_row['total_params'],
            'parameters_millions': batch1_row['total_params'] / 1e6,
            'efficiency_img_per_sec_per_million_params': batch1_row['images_per_second'] / (batch1_row['total_params'] / 1e6)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('single_image_throughput', ascending=False)
    
    # Save to CSV
    comparison_df.to_csv(output_file, index=False, float_format='%.2f')
    print(f"\n📄 Comparison CSV saved to: {output_file}")
    
    return comparison_df

def main():
    parser = argparse.ArgumentParser(description='Analyze inference speed results')
    parser.add_argument('--results_dir', default='inference_results', 
                       help='Directory containing inference speed results')
    parser.add_argument('--csv_file', help='Specific CSV file to analyze')
    parser.add_argument('--save_plots', action='store_true', 
                       help='Save plots to file')
    parser.add_argument('--no_plots', action='store_true', 
                       help='Skip creating plots')
    
    args = parser.parse_args()
    
    # Load data
    if args.csv_file:
        if not os.path.exists(args.csv_file):
            print(f"CSV file '{args.csv_file}' not found!")
            return
        df = pd.read_csv(args.csv_file)
        csv_file = args.csv_file
        print(f"Loading results from: {csv_file}")
    else:
        result = load_latest_results(args.results_dir)
        if result is None:
            return
        df, csv_file = result
    
    print(f"\nLoaded {len(df)} inference measurements for {df['backbone'].nunique()} different backbones")
    
    # Create summary
    create_summary_table(df)
    
    # Create comparison CSV
    create_comparison_csv(df)
    
    # Create plots
    if not args.no_plots:
        try:
            plot_inference_speeds(df, save_plots=args.save_plots)
        except Exception as e:
            print(f"Error creating plots: {e}")
            print("You may need to install matplotlib and seaborn: pip install matplotlib seaborn")

if __name__ == '__main__':
    main() 