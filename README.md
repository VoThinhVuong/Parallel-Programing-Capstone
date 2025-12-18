# CIFAR-10 CNN with Autoencoder - Parallel Programming Project

**CSC14120 Parallel Programming Final Project (2025)**

## Project Overview

This project implements a CNN for CIFAR-10 classification with an autoencoder-style decoder for joint training. The system demonstrates GPU acceleration through multiple CUDA implementations, progressing from naive parallelization to advanced memory optimizations.

### Implementations

1. **CPU Baseline** (`CPU/`) - Sequential C implementation with joint training
2. **GPU Naive** (`GPU_naive/`) - Direct CUDA port with basic kernels
3. **GPU Shared Memory** (`GPU_v2_shared_mem/`) - Optimized with shared memory
4. **GPU Optimized** (`GPU_v3_optimized_gradient/`) - Advanced gradient optimizations

## Architecture

**Encoder (Feature Extraction):**
```
Input: 32×32×3 → Conv1(32 filters) → Pool1 → Conv2(128 filters) → Pool2 → 8×8×128
```

**Classifier:**
```
8×8×128 (8192 features) → FC1(128) → ReLU → FC2(10) → Softmax
```

**Decoder (Reconstruction):**
```
8×8×128 → Upsample(16×16) → TransConv1(64 filters) → Upsample(32×32) → TransConv2(3 filters) → 32×32×3
```

**Joint Training:** Both classification and reconstruction losses update encoder weights through gradient accumulation.

## Requirements

### Hardware
- **CPU Version**: Any x64 processor, 2GB RAM
- **GPU Versions**: 
  - NVIDIA GPU (Compute Capability 5.0+)
  - 8GB+ GPU memory (16GB recommended)
  - Tested on: RTX 3050 4GB, Tesla T4

### Software
- **Compiler**: GCC 7.0+ (CPU), NVCC from CUDA Toolkit (GPU)
- **Build Tools**: GNU Make 3.5+
- **CUDA Toolkit**: 10.0 or later (12.x recommended)
- **Python**: 3.7+ with matplotlib, numpy (for visualization)
- **OS**: Linux (preferred) or Windows with CUDA support

*This project has mostly been tested and executed on Google Colab*

### Dataset
Download **CIFAR-10 binary version** from https://www.cs.toronto.edu/~kriz/cifar.html

Place in project root:
```
cifar-10-batches-bin/
├── data_batch_1.bin ... data_batch_5.bin
├── test_batch.bin
└── batches.meta.txt
```

## Compilation commands

### 1. Verify Setup
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Check dataset
ls cifar-10-batches-bin/
```

### 2. Compilation

**CPU Version:**
```bash
cd CPU
make clean && make
```

**GPU:**
```bash
cd GPU_<version>
make clean && make
```

## Execution commands

### 1. Training

**Basic training (20 epochs):**
* Default: 20 epochs, lr=0.01, batch_size=64 (32 for cpu), 5 training batches

```bash
cd GPU_<version>
./cifar10_cnn_gpu.exe        # Windows
./cifar10_cnn_gpu            # Linux
```

**Custom arguments:**
```bash
cd GPU_<version>
# Usage: ./cifar10_cnn_gpu [num_epochs] [learning_rate] [batch_size] [num_train_batches]

./cifar10_cnn_gpu 20 0.01 16 3        # 20 epochs, lr=0.01, batch_size=16, use 3 training batches
./cifar10_cnn_gpu 10 0.01 64 1        # Quick test: 10 epochs, batch_size=64, only 1st training batch
./cifar10_cnn_gpu 50 0.001 128 5      # 50 epochs, lr=0.001, batch_size=128, all 5 training batches
```

### 2. Feature Extraction & SVM Classification

**Extract features:**
```bash
cd GPU_<version>                 # Or any GPU version
./extract_features.exe
```

**Train SVM (requires cuML):**
```bash
cd SVM_cuML

# Basic training (default: RBF kernel, C=10, gamma=auto)
python train_svm_cuml.py

# Custom hyperparameters
python train_svm_cuml.py --C 1.0 --gamma 0.001 --kernel rbf

# Linear kernel
python train_svm_cuml.py --kernel linear --C 5.0

# Quick test with subset (first 1000 samples)
python train_svm_cuml.py --subset 1000

# Custom paths
python train_svm_cuml.py --train-features ../extracted_features/train_features_v3.bin \
                         --output ./models/my_svm_model.pkl

# Evaluate model
python evaluate_svm_cuml.py

# Custom model and data paths
python evaluate_svm_cuml.py --model ./models/my_svm_model.pkl \
                            --test-features ../extracted_features/test_features_v3.bin \

# Custom output paths
python evaluate_svm_cuml.py --output-matrix ./results/confusion_matrix.png \
                            --output-results ./results/eval_results.txt

# Skip plotting (faster)
python evaluate_svm_cuml.py --no-plot

# Quick test with subset
python evaluate_svm_cuml.py --subset 500
```

## Expected Outputs

### Training Output
```
=== CIFAR-10 CNN Training (GPU Naive Implementation) ===
Batch size: 64
Learning rate: 0.001
Number of epochs: 20

Using GPU: NVIDIA GeForce RTX 3070
Compute Capability: 8.6
Global Memory: 8.00 GB

Creating CNN model on GPU...
Creating decoder...
Weights initialized and copied to GPU

Model Architecture:
  Input: 32x32x3
  Conv1: 32 filters, 3x3 kernel -> 32x32x32
  Pool1: 2x2 -> 16x16x32
  Conv2: 128 filters, 3x3 kernel -> 16x16x128
  Pool2: 2x2 -> 8x8x128
  FC1: 8192 -> 128
  FC2: 128 -> 10 (output)
  Decoder: Pool2 -> Upsample -> TransConv(64ch) -> Upsample -> TransConv(3ch) -> 32x32x3

Epoch 1/20:
  Progress: [========================================] 781/781 (100.0%)
  Class Loss: 2.1234, Acc: 0.1567, Recon Loss: 0.012345
  Epoch completed in 12.34 seconds
Evaluating on 10000 samples...
Test Loss: 2.0456, Test Accuracy: 0.1823

...

=== Training Complete ===
Total training time: 246.80 seconds
Average time per epoch: 12.34 seconds
Encoder weights saved to 'encoder_weights.bin'
```

### Reconstruction Visualization
```bash
# Generate comparison plots (requires Python)
python visualize_reconstructions.py --file extracted_features/reconstructed_images_gpu.bin
```

Outputs:
- `reconstruction_comparison.png` - Side-by-side original vs reconstructed

## Configuration

### Training Hyperparameters
- **Batch size**: 64 (GPU), 32 (CPU)
- **Learning rate**: 0.01 (reduced for joint training stability)
- **Epochs**: 20 (default, configurable via command-line)
- **Optimizer**: SGD with gradient accumulation
- **Loss functions**: 
  - Classification: Cross-entropy
  - Reconstruction: MSE

### Makefile Configuration
Update `arch=sm_XX` in Makefile for your GPU:
- `sm_50`: Maxwell (GTX 9xx)
- `sm_60`: Pascal (GTX 10xx)
- `sm_75`: Turing (RTX 20xx)
- `sm_86`: Ampere (RTX 30xx)


## Compilation Details

### Makefile Targets
```bash
make              # Build both training and feature extraction
make clean        # Remove all build artifacts
make run          # Build and run training
make extract      # Build and run feature extraction
```

### Manual Compilation (if needed)
```bash
# GPU version
nvcc -O3 -arch=sm_75 main.cu cnn.cu forward.cu backward.cu decoder.cu \
     feature_extractor.cu data_loader.cu -o cifar10_cnn_gpu -lcudart

# CPU version
gcc -O3 -std=c99 main.c cnn.c forward.c backward.c data_loader.c \
    -o cifar10_cnn_cpu -lm
```

## Troubleshooting

### CUDA Errors
- **Out of memory**: Reduce batch size or close GPU applications
- **No device found**: Check `nvidia-smi` and update drivers
- **Compilation fails**: Verify CUDA toolkit (`nvcc --version`) and correct compute capability in Makefile

### Training Issues
- **NaN losses**: Lower learning rate (0.001 recommended), check gradient zeroing
- **Slow convergence**: Normal for 10-20 epochs; increase to 50+ for better accuracy
- **Data loading errors**: Ensure CIFAR-10 binary files in `cifar-10-batches-bin/`

### Feature Extraction
- **Classifier doesn't run**: Check that labels are saved correctly, verify file permissions
- **SVM training fails**: Install cuML dependencies (`pip install -r SVM_cuML/requirements.txt`)

## Performance Notes

**GPU Speedup**: Expect 10-50× faster than CPU depending on implementation:
- Naive: ~10-15× speedup
- Shared Memory (v2): ~20-30× speedup  
- Optimized (v3): ~30-50× speedup

**Training Time** (20 epochs, GTX 1060):
- CPU: ~2-3 hours
- GPU Naive: ~10-15 minutes
- GPU Optimized: ~5-8 minutes

## Project Goals

This project demonstrates parallel programming concepts for deep learning:
1. **Encoder-Decoder Architecture**: Joint training with classification and reconstruction
2. **CUDA Parallelization**: Progressive GPU optimizations (naive → shared memory → advanced)
3. **Memory Management**: Efficient device memory allocation and transfer
4. **Feature Extraction**: Export learned representations for downstream tasks
5. **Performance Analysis**: Speedup measurement and profiling

## License

Academic project for CSC14120 Parallel Programming (2025)

## References

- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- CUDA Programming Guide: https://docs.nvidia.com/cuda/
- cuDNN Documentation: https://docs.nvidia.com/deeplearning/cudnn/
