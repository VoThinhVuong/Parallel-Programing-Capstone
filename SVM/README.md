# SVM Classification Using ThunderSVM

This folder implements Steps 4 & 5 of the CIFAR-10 CNN project: Training and evaluating an SVM classifier using features extracted from the GPU-accelerated CNN encoder.

## Overview

After the CNN encoder extracts features from images (completed in `GPU_naive` folder), we use **ThunderSVM** - a GPU-accelerated SVM library - to train a classifier and achieve 60-65% accuracy on CIFAR-10.

### What is ThunderSVM?

[ThunderSVM](https://github.com/Xtra-Computing/thundersvm) is a fast GPU-based implementation of Support Vector Machines (SVMs). It provides:
- GPU acceleration using CUDA
- Support for multi-class classification
- Multiple kernel functions (RBF, Linear, Polynomial, Sigmoid)
- Compatibility with scikit-learn API

## Pipeline

```
GPU_naive/train_features.bin (50,000 × 8,192)  ─┐
GPU_naive/train_labels.bin   (50,000)          ─┤
                                                 ├─> Step 4: Train SVM ─> svm_model.pkl
GPU_naive/test_features.bin  (10,000 × 8,192)  ─┤
GPU_naive/test_labels.bin    (10,000)          ─┘
                                                 └─> Step 5: Evaluate  ─> Results
```

## Prerequisites

1. **Completed GPU_naive feature extraction** - Ensure these files exist:
   - `../GPU_naive/train_features.bin`
   - `../GPU_naive/train_labels.bin`
   - `../GPU_naive/test_features.bin`
   - `../GPU_naive/test_labels.bin`

2. **Python 3.7+** with CUDA support

3. **CUDA Toolkit** (10.0 or higher)

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `thundersvm` - GPU-accelerated SVM library
- `numpy` - Numerical computing
- `scikit-learn` - Evaluation metrics
- `matplotlib` - Visualization
- `seaborn` - Enhanced plotting

### 2. Verify ThunderSVM Installation

```bash
python -c "from thundersvm import SVC; print('ThunderSVM installed successfully!')"
```

If you encounter issues, install ThunderSVM manually:
```bash
# Using pip
pip install thundersvm

# Or build from source (for latest version)
git clone https://github.com/Xtra-Computing/thundersvm.git
cd thundersvm
mkdir build
cd build
cmake ..
make
pip install python/
```

## Usage

### Step 4: Train SVM

Train the SVM classifier with RBF kernel:

```bash
python train_svm.py
```

**Hyperparameters:**
- Kernel: RBF (Radial Basis Function)
- C: 10.0 (Regularization parameter)
- gamma: auto (1 / n_features = 1/8192)

**Output:**
- `svm_model.pkl` - Trained SVM model (saved using pickle)
- Training accuracy printed to console

**Expected Runtime:** 5-15 minutes (depending on GPU)

### Step 5: Evaluate SVM

Evaluate the trained model on test data:

```bash
python evaluate_svm.py
```

**Output:**
- Test accuracy (expected: 60-65%)
- Per-class accuracy for all 10 CIFAR-10 classes
- Confusion matrix visualization (`confusion_matrix.png`)
- Detailed results (`evaluation_results.txt`)

## Files in This Folder

```
SVM/
├── train_svm.py              # Step 4: Train SVM classifier
├── evaluate_svm.py           # Step 5: Evaluate and generate metrics
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── svm_model.pkl            # Trained model (generated)
├── evaluation_results.txt   # Evaluation metrics (generated)
└── confusion_matrix.png     # Confusion matrix plot (generated)
```

## Feature Format

The binary feature files use the following format:

```
[4 bytes: num_samples (int32)] [features array (float32)]
```

- **train_features.bin**: 50,000 samples × 8,192 features
- **test_features.bin**: 10,000 samples × 8,192 features
- **train_labels.bin**: 50,000 labels (int32, range 0-9)
- **test_labels.bin**: 10,000 labels (int32, range 0-9)

## Expected Results

### Training Phase
```
Training samples: 50,000
Feature dimension: 8,192
Training time: ~5-15 minutes
Training accuracy: ~65-70%
```

### Evaluation Phase
```
Test Accuracy: 60-65%
Per-class accuracy: ~50-70% (varies by class)
Inference time: ~1-2 seconds for 10,000 samples
```

### Sample Output

```
==============================================================
EVALUATION RESULTS
==============================================================

Test Accuracy: 62.45%
Expected Range: 60-65%

--------------------------------------------------------------
Per-Class Accuracy:
--------------------------------------------------------------
airplane    : 68.20% ( 682/1000)
automobile  : 72.30% ( 723/1000)
bird        : 51.80% ( 518/1000)
cat         : 45.60% ( 456/1000)
deer        : 58.90% ( 589/1000)
dog         : 54.20% ( 542/1000)
frog        : 69.80% ( 698/1000)
horse       : 66.50% ( 665/1000)
ship        : 73.10% ( 731/1000)
truck       : 64.00% ( 640/1000)
```

## CIFAR-10 Classes

The classifier predicts 10 classes:
1. airplane
2. automobile
3. bird
4. cat
5. deer
6. dog
7. frog
8. horse
9. ship
10. truck

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** ThunderSVM uses GPU memory. If you encounter OOM errors:
- Close other GPU applications
- Reduce batch size in ThunderSVM (modify `train_svm.py`)
- Use a smaller subset of training data for testing

### Issue: "Module 'thundersvm' not found"
**Solution:** 
```bash
pip uninstall thundersvm
pip install thundersvm
```

### Issue: "Features file not found"
**Solution:** Make sure you've run the feature extraction in `GPU_naive` folder:
```bash
cd ../GPU_naive
make extract
cd ../SVM
```

### Issue: Low accuracy (<50%)
**Possible causes:**
- Features not properly normalized
- Incorrect hyperparameters
- Feature extraction errors in GPU_naive

**Solution:** Check feature extraction output and verify file sizes match expected dimensions.

## Performance Notes

### Why 60-65% Accuracy?

This is expected for a **linear classifier on CNN features**:
- The CNN encoder extracts spatial features (8×8×128 = 8,192 dimensions)
- SVM with RBF kernel acts as a non-linear classifier on these features
- Full end-to-end CNN training would achieve ~70-80% with data augmentation
- 60-65% is competitive for feature-based transfer learning without fine-tuning

### Comparison with Other Methods

| Method | Test Accuracy |
|--------|--------------|
| Random Baseline | ~10% |
| Logistic Regression on Features | ~55-60% |
| **SVM (RBF) on Features** | **60-65%** |
| Fine-tuned CNN | ~70-80% |
| State-of-the-art (ResNet, etc.) | ~95%+ |

## References

- **ThunderSVM Paper:** Wen, Z., et al. "ThunderSVM: A Fast SVM Library on GPUs and CPUs." JMLR 2018.
- **GitHub:** https://github.com/Xtra-Computing/thundersvm
- **CIFAR-10 Dataset:** https://www.cs.toronto.edu/~kriz/cifar.html

## Next Steps

After completing SVM training and evaluation:
1. Experiment with different hyperparameters (C, gamma)
2. Try different kernels (linear, polynomial)
3. Compare with other classifiers (Random Forest, XGBoost)
4. Analyze misclassified examples
5. Fine-tune the CNN encoder for better features

## License

This implementation is part of the CSC14120 Parallel Programming Final Project.
