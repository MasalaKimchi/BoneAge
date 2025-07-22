# Cyclical Learning Rate Training Script

This document describes the enhanced training script `train_advanced_kjc.py` which incorporates cyclical learning rate functionality for improved model training.

## Overview

Cyclical Learning Rate (CLR) is a training technique that varies the learning rate between minimum and maximum boundaries in cycles. This approach can help models:

- Escape local minima and saddle points
- Converge faster than fixed learning rates
- Achieve better final accuracy
- Reduce the need for extensive learning rate tuning

## Files Added

- `train_advanced_kjc.py` - Enhanced training script with CLR support
- `example_cyclical_lr.py` - Example usage script
- `CYCLICAL_LR_README.md` - This documentation

## Usage

### Basic Command with Cyclical LR

```bash
python train_advanced_kjc.py \
    --lr 5e-5 \
    --backbone resnet101v2 \
    --model_type attn_sex \
    --fine_tune \
    --epochs_frozen 20 \
    --epochs_finetune 80 \
    --use_cyclical_lr \
    --clr_max_lr 5e-3 \
    --clr_step_size 1000 \
    --clr_mode triangular
```

### New Command Line Arguments

#### Cyclical Learning Rate Parameters

- `--use_cyclical_lr` - Enable cyclical learning rate (default: False)
- `--clr_max_lr FLOAT` - Maximum learning rate for cycles (default: 1e-2)
- `--clr_step_size INT` - Number of iterations for half a cycle (default: 2000)
- `--clr_mode STR` - Cycling mode: triangular, triangular2, exp_range (default: triangular)
- `--clr_gamma FLOAT` - Gamma value for exp_range mode (default: 1.0)

#### Existing Parameters (still available)

All original parameters from `train_advanced.py` are preserved:
- `--lr` - Base learning rate (used as minimum LR for CLR)
- `--backbone` - Model backbone architecture
- `--model_type` - Model type (baseline, sex, attn_sex)
- `--fine_tune` - Enable fine-tuning
- `--epochs_frozen` - Epochs with frozen backbone
- `--epochs_finetune` - Epochs for fine-tuning
- And all other original parameters...

## Cyclical Learning Rate Modes

### 1. Triangular Mode
```
LR = base_lr + (max_lr - base_lr) * max(0, 1 - |x|)
```
- Basic triangular wave cycling
- Constant amplitude throughout training
- Good starting point for most experiments

### 2. Triangular2 Mode
```
LR = base_lr + (max_lr - base_lr) * max(0, 1 - |x|) / 2^(cycle-1)
```
- Amplitude decreases by half each cycle
- Provides more exploration early, refinement later
- Good for longer training runs

### 3. Exponential Range Mode
```
LR = base_lr + (max_lr - base_lr) * max(0, 1 - |x|) * gamma^iterations
```
- Amplitude decreases exponentially
- Controlled by gamma parameter (should be < 1)
- Most aggressive decay option

## Output Files

When using cyclical LR, output files include a "_clr" suffix:

- `{model}_{backbone}_clr_{timestamp}.h5` - Model weights
- `{model}_{backbone}_clr_{timestamp}.json` - Training results with CLR history
- `{model}_{backbone}_clr_{timestamp}_predictions.csv` - Test predictions
- `{model}_{backbone}_clr_{timestamp}_lr_schedule.png` - Learning rate visualization

## Examples

### Example 1: Your Original Request
```bash
python train_advanced_kjc.py \
    --lr 5e-5 \
    --backbone resnet101v2 \
    --model_type attn_sex \
    --fine_tune \
    --epochs_frozen 20 \
    --epochs_finetune 80 \
    --use_cyclical_lr \
    --clr_max_lr 5e-3 \
    --clr_step_size 1000
```

### Example 2: Conservative CLR for Baseline Model
```bash
python train_advanced_kjc.py \
    --lr 1e-4 \
    --backbone resnet50 \
    --model_type baseline \
    --epochs_frozen 10 \
    --use_cyclical_lr \
    --clr_max_lr 1e-3 \
    --clr_step_size 1500 \
    --clr_mode triangular2
```

### Example 3: Aggressive CLR with Exponential Decay
```bash
python train_advanced_kjc.py \
    --lr 1e-5 \
    --backbone xception \
    --model_type sex \
    --fine_tune \
    --use_cyclical_lr \
    --clr_max_lr 1e-2 \
    --clr_step_size 800 \
    --clr_mode exp_range \
    --clr_gamma 0.9999
```

## Parameter Recommendations

### Learning Rate Range
- **Base LR**: Start with your normal learning rate (e.g., 1e-4, 5e-5)
- **Max LR**: Typically 10-100x the base LR
- **Safe ratio**: max_lr = base_lr * 10 to 20
- **Aggressive ratio**: max_lr = base_lr * 50 to 100

### Step Size
- **Rule of thumb**: 2-10 epochs worth of iterations
- **Formula**: step_size = epochs_per_cycle * steps_per_epoch / 2
- **Conservative**: 2000-4000 iterations
- **Aggressive**: 500-1000 iterations

### Mode Selection
- **Start with**: `triangular` for initial experiments
- **Long training**: `triangular2` for runs >50 epochs
- **Fine control**: `exp_range` with gamma ~0.9999

## Running the Examples

Use the provided example script:

```bash
python example_cyclical_lr.py
```

This interactive script will show you different CLR configurations and let you choose which one to run.

## Monitoring Training

The script automatically:

1. **Removes conflicting callbacks** - ReduceLROnPlateau is disabled when using CLR
2. **Saves LR history** - All learning rate values and iterations are logged
3. **Generates plots** - Learning rate schedule is visualized and saved
4. **Resets for fine-tuning** - CLR cycles restart when entering fine-tuning phase

## Benefits of Cyclical Learning Rate

1. **Escapes Local Minima**: Higher learning rates help jump out of poor local optima
2. **Faster Convergence**: Can reach good solutions quicker than fixed LR
3. **Better Generalization**: Oscillations can improve final model performance
4. **Reduced Tuning**: Less need to find the "perfect" learning rate
5. **Automatic Scheduling**: No manual LR decay schedule required

## Tips for Success

1. **Start Conservative**: Begin with smaller max_lr and larger step_size
2. **Monitor Closely**: Watch the learning rate plot and loss curves
3. **Experiment**: Try different modes and parameters for your specific dataset
4. **Use with Fine-tuning**: CLR works well with the two-phase training approach
5. **Compare Results**: Train identical models with and without CLR to see benefits

## Troubleshooting

### Training Unstable
- Reduce `clr_max_lr`
- Increase `clr_step_size`
- Switch to `triangular2` mode

### No Improvement Over Fixed LR
- Increase `clr_max_lr` (within reason)
- Decrease `clr_step_size`
- Try different cycling modes

### Loss Spikes Too High
- Your max_lr might be too aggressive
- Consider triangular2 or exp_range modes
- Check if gradient clipping is enabled

## References

- Smith, L. N. (2017). "Cyclical Learning Rates for Training Neural Networks"
- Paper: https://arxiv.org/abs/1506.01186
- Implementation inspired by TensorFlow/Keras documentation 