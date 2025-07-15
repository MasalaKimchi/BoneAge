"""
Example script showing how to run inference speed measurements
for different backbone architectures.
"""

import subprocess
import sys

def run_command(cmd):
    """Run a command and print its output."""
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    """Run different inference speed measurement scenarios."""
    
    print("🚀 Running Inference Speed Measurements")
    print("This will test different backbone architectures for inference speed")
    
    # Example 1: Test all backbones with default settings
    print("\n📊 Example 1: Testing all backbones with default settings")
    success = run_command("python compute_inference_speed.py")
    if not success:
        print("❌ Failed to run basic inference speed test")
        return
    
    # Example 2: Test specific fast backbones
    print("\n⚡ Example 2: Testing fast backbones (MobileNet variants)")
    cmd = "python compute_inference_speed.py --backbones mobilenet mobilenet_v2 mobilenet_v3_small mobilenet_v3_large"
    run_command(cmd)
    
    # Example 3: Test with different batch sizes
    print("\n📈 Example 3: Testing with different batch sizes")
    cmd = "python compute_inference_speed.py --backbones xception resnet50 --batch_sizes 1 8 16 64"
    run_command(cmd)
    
    # Example 4: Quick test with fewer runs
    print("\n⏱️ Example 4: Quick test with fewer runs")
    cmd = "python compute_inference_speed.py --backbones xception convnext_tiny --num_runs 5 --batch_sizes 1 4"
    run_command(cmd)
    
    print("\n✅ All inference speed measurements completed!")
    print("📁 Check the 'inference_results' directory for CSV files")

if __name__ == '__main__':
    main() 