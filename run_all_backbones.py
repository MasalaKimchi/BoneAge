import subprocess
import sys
import os
import json
from modeling_advanced import BACKBONE_MAP

# Arguments for train_advanced.py
EPOCHS_FROZEN = 20
EPOCHS_FINETUNE = 0
USE_CLAHE = True
RESULTS_DIR = 'backbone_results'

os.makedirs(RESULTS_DIR, exist_ok=True)

# Path to Python executable in venv
PYTHON_EXEC = os.path.join('venv', 'Scripts', 'python.exe')

# List of backbones
backbones = list(BACKBONE_MAP.keys())

for backbone in backbones:
    print(f'\nRunning backbone: {backbone}')
    result_file = os.path.join(RESULTS_DIR, f'{backbone}_performance.json')
    # Build command
    cmd = [
        PYTHON_EXEC, 'train_advanced.py',
        '--epochs_frozen', str(EPOCHS_FROZEN),
        '--epochs_finetune', str(EPOCHS_FINETUNE),
        '--backbone', backbone,
        '--use_clahe_val_test'
    ]
    # Run the training script and capture output
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = proc.stdout + proc.stderr
        # Parse best validation MAE and epoch from output
        best_mae = None
        best_epoch = None
        for line in output.splitlines():
            if 'Best validation MAE:' in line:
                # Example: Best validation MAE: 5.1234 at epoch 17
                parts = line.strip().split()
                best_mae = float(parts[3])
                best_epoch = int(parts[-1])
                break
        # Save results
        with open(result_file, 'w') as f:
            json.dump({
                'backbone': backbone,
                'best_val_mae': best_mae,
                'best_epoch': best_epoch,
                'output': output
            }, f, indent=2)
        print(f'Results saved to {result_file}')
    except subprocess.CalledProcessError as e:
        print(f'Error running backbone {backbone}:')
        print(e.stdout)
        print(e.stderr)
        with open(result_file, 'w') as f:
            json.dump({
                'backbone': backbone,
                'error': True,
                'output': e.stdout + e.stderr
            }, f, indent=2) 