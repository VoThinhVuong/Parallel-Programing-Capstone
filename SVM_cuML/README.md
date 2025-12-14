# SVM Training and Evaluation with cuML (GPU-Accelerated)

This folder contains GPU-accelerated SVM training and evaluation using NVIDIA's RAPIDS cuML library for the CIFAR-10 image classification pipeline.

## Overview

This implementation uses cuML's GPU-accelerated SVM to train on features extracted from a CNN. It provides significant speedup compared to CPU-based implementations while maintaining high accuracy.

### Key Features
- **GPU-Accelerated**: Uses cuML for fast SVM training and inference on NVIDIA GPUs
- **RBF Kernel**: Radial Basis Function kernel with configurable hyperparameters
- **High Accuracy**: Expected accuracy of 60-65% on CIFAR-10 test set
- **Comprehensive Evaluation**: Includes confusion matrix, per-class accuracy, and performance metrics

## Requirements

### System Requirements
- NVIDIA GPU with CUDA support (Compute Capability 7.0+)
- CUDA Toolkit 12.x
- Python 3.10, 3.11, or 3.12
- 8GB+ GPU memory recommended

### Software Dependencies
- RAPIDS cuML 25.12
- cuDF 25.12
- CuPy
- NumPy, Matplotlib, Seaborn, scikit-learn

## Installation

### Step 1: Install RAPIDS cuML

First, ensure you're in a virtual environment or conda environment:

```bash
# Using make (recommended)
make install-pip

# Or manually
pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12==25.12.* cudf-cu12==25.12.*
```

### Step 2: Verify Installation

```bash
make verify
```

Expected output:
```
Success! Installed cuML: 25.12.x | cuDF: 25.12.x
```

### Step 3: Install Additional Dependencies

```bash
make install-deps
```

This installs NumPy, Matplotlib, Seaborn, scikit-learn, and other required packages.

## Usage

### Prerequisites

Before training the SVM, you need extracted features from the CNN:

1. Navigate to one of the GPU implementations (e.g., `GPU_naive/`, `GPU_v2_shared_mem/`, or `GPU_v3_optimized_gradient/`)
2. Run feature extraction:
   ```bash
   cd ../GPU_naive
   make extract-features
   ```
3. This creates feature files in `../extracted_features/`:
   - `train_features.bin` (50,000 samples)
   - `train_labels.bin`
   - `test_features.bin` (10,000 samples)
   - `test_labels.bin`

### Training the SVM

#### Basic Training (Default Parameters)
```bash
make train
```

Parameters:
- Kernel: RBF (Radial Basis Function)
- C: 10.0
- Gamma: auto (1/n_features)

#### Custom Training
```bash
# Custom hyperparameters
python train_svm_cuml.py --C 10.0 --gamma 0.001 --kernel rbf

# Different kernel
python train_svm_cuml.py --kernel linear --C 1.0

# Specify custom paths
python train_svm_cuml.py --train-features ../extracted_features/train_features.bin \
                         --train-labels ../extracted_features/train_labels.bin \
                         --output my_svm_model.pkl
```

#### Quick Test Training (Subset)
For quick testing without waiting for full training:
```bash
make train-test  # Uses 1000 samples
```

### Evaluating the SVM

#### Basic Evaluation
```bash
make evaluate
```

This will:
- Load the trained model
- Predict on test features
- Calculate accuracy and metrics
- Generate confusion matrix plot
- Save results to text file

#### Quick Test Evaluation
```bash
make evaluate-test  # Uses 500 samples
```

#### Custom Evaluation
```bash
# Custom paths
python evaluate_svm_cuml.py --test-features ../extracted_features/test_features.bin \
                            --test-labels ../extracted_features/test_labels.bin \
                            --model svm_model_cuml.pkl \
                            --output-matrix my_confusion_matrix.png
```

### Full Pipeline

Run both training and evaluation in sequence:

```bash
# Full pipeline
make pipeline

# Quick test pipeline (with subsets)
make test-pipeline
```

## Output Files

After training and evaluation, the following files are generated:

1. **svm_model_cuml.pkl** - Trained cuML SVM model (can be reused)
2. **confusion_matrix_cuml.png** - Confusion matrix heatmap visualization
3. **evaluation_results_cuml.txt** - Detailed evaluation metrics

## Expected Results

### Accuracy
- **Target**: 60-65%
- **Typical**: ~62-63%

### Performance
- **Training**: ~10-30 seconds on modern GPU (depends on GPU model)
- **Inference**: ~0.1-0.5 seconds for 10,000 test samples
- **Speedup**: 10-50x faster than CPU implementations

### Per-Class Performance
Some classes perform better than others:
- **Good**: airplane, automobile, ship, truck (vehicles)
- **Challenging**: cat, dog, bird (similar features)

## Makefile Commands

### Installation
```bash
make install-pip      # Install RAPIDS cuML suite
make verify           # Verify installation
make install-deps     # Install additional dependencies
```

### Training
```bash
make train            # Train with default parameters
make train-custom     # Train with custom parameters
make train-test       # Quick training on subset
```

### Evaluation
```bash
make evaluate         # Evaluate on full test set
make evaluate-test    # Quick evaluation on subset
```

### Pipeline
```bash
make pipeline         # Full train + evaluate
make test-pipeline    # Quick test pipeline
```

### Utilities
```bash
make clean            # Remove generated files
make clean-pip        # Uninstall RAPIDS packages
make help             # Show RAPIDS installation help
make help-svm         # Show SVM commands help
```

## Hyperparameter Tuning

### C Parameter (Regularization)
Controls the trade-off between margin maximization and training error:
- **Low C (0.1-1)**: Larger margin, simpler model, may underfit
- **Medium C (10)**: **Recommended starting point**
- **High C (100-1000)**: Smaller margin, complex model, may overfit

```bash
python train_svm_cuml.py --C 1.0    # Simpler model
python train_svm_cuml.py --C 10.0   # Default
python train_svm_cuml.py --C 100.0  # More complex
```

### Gamma Parameter (RBF Kernel)
Controls the influence of a single training example:
- **Low gamma (0.0001)**: Far reach, smoother decision boundary
- **auto (1/n_features)**: **Recommended default**
- **High gamma (1.0)**: Close reach, complex decision boundary

```bash
python train_svm_cuml.py --gamma 0.001  # Smoother
python train_svm_cuml.py --gamma auto   # Default
python train_svm_cuml.py --gamma 0.01   # More complex
```

### Kernel Selection
```bash
python train_svm_cuml.py --kernel rbf     # Non-linear (default, better accuracy)
python train_svm_cuml.py --kernel linear  # Linear (faster, lower memory)
```

## Troubleshooting

### GPU Memory Issues
If you encounter out-of-memory errors:

1. Use a subset for training:
   ```bash
   python train_svm_cuml.py --subset 10000
   ```

2. Use linear kernel (lower memory):
   ```bash
   python train_svm_cuml.py --kernel linear
   ```

3. Monitor GPU memory:
   ```bash
   nvidia-smi
   ```

### CUDA/cuML Issues
If cuML fails to import:

1. Check CUDA installation:
   ```bash
   nvidia-smi
   ```

2. Verify Python version (must be 3.10, 3.11, or 3.12):
   ```bash
   python --version
   ```

3. Reinstall RAPIDS:
   ```bash
   make clean-pip
   make install-pip
   ```

### Missing Features
If feature files are not found:

1. Make sure features are extracted:
   ```bash
   cd ../GPU_naive
   make extract-features
   ```

2. Check that feature files exist:
   ```bash
   ls -lh ../extracted_features/
   ```

## Comparison with Other SVM Implementations

| Implementation | Library | Hardware | Training Time | Accuracy |
|---------------|---------|----------|---------------|----------|
| **cuML** | RAPIDS cuML | GPU | ~10-30s | 60-65% |
| ThunderSVM | ThunderSVM | GPU | ~20-60s | 60-65% |
| scikit-learn | libsvm | CPU | ~5-15 min | 60-65% |

**Advantages of cuML:**
- Faster training and inference
- Seamless integration with GPU workflow
- Better memory management
- Part of RAPIDS ecosystem

## Advanced Usage

### Python API

You can also use the modules directly in Python:

```python
from train_svm_cuml import load_features, train_svm_cuml, save_model
from evaluate_svm_cuml import evaluate_svm_cuml

# Load data
train_features, train_labels = load_features(
    '../extracted_features/train_features.bin',
    '../extracted_features/train_labels.bin'
)

# Train
model, training_time = train_svm_cuml(train_features, train_labels, C=10.0, gamma='auto')

# Save
save_model(model, 'my_model.pkl')

# Evaluate
test_features, test_labels = load_features(
    '../extracted_features/test_features.bin',
    '../extracted_features/test_labels.bin'
)
predictions, accuracy, conf_matrix, pred_time = evaluate_svm_cuml(
    model, test_features, test_labels
)

print(f"Accuracy: {accuracy:.2f}%")
```

## References

- [RAPIDS cuML Documentation](https://docs.rapids.ai/api/cuml/stable/)
- [cuML SVM Guide](https://docs.rapids.ai/api/cuml/stable/api.html#support-vector-machines)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## License

This project is part of the Parallel Programming Capstone course.

## Support

For issues specific to:
- cuML installation: Check [RAPIDS Installation Guide](https://rapids.ai/start.html)
- SVM training: Review hyperparameter settings
- GPU memory: Try subset training or linear kernel
