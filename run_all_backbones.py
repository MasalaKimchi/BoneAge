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
        # Find the most recent JSON file for this backbone
        json_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith(f'{backbone}_') and f.endswith('.json')]
        if not json_files:
            print(f'No result JSON file found for backbone {backbone}!')
            continue
        # Sort by date in filename (YYYYMMDD)
        json_files.sort(reverse=True)
        result_file = os.path.join(RESULTS_DIR, json_files[0])
        print(f'Results saved to {result_file}')
    except subprocess.CalledProcessError as e:
        print(f'Error running backbone {backbone}:')
        print(e.stdout)
        print(e.stderr) 