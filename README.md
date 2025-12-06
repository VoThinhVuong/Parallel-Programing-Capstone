# CIFAR-10 CNN Training - Parallel Programming Project

CSC14120 Parallel Programming Final Project (2025)

## Project Overview

This project implements a Convolutional Neural Network (CNN) for CIFAR-10 image classification with three different implementations to demonstrate the impact of parallel computing and GPU acceleration.

### Implementations

1. **CPU Baseline** (`CPU/`) - Sequential C implementation
2. **GPU Naive** (`GPU_naive/`) - Direct CUDA port without optimizations
3. **GPU Optimized** (`GPU_optimized/`) - Optimized CUDA with cuDNN/cuBLAS *(To be implemented)*

## CNN Architecture

```
Input: 32×32×3 RGB images
├─ Conv1: 32 filters, 3×3 kernel, stride 1, padding 1 → 32×32×32
├─ ReLU activation
├─ MaxPool1: 2×2, stride 2 → 16×16×32
├─ Conv2: 64 filters, 3×3 kernel, stride 1, padding 1 → 16×16×64
├─ ReLU activation
├─ MaxPool2: 2×2, stride 2 → 8×8×64
├─ FC1: 4096 → 128 neurons
├─ ReLU activation
├─ FC2: 128 → 10 classes
└─ Softmax
```

**Total Parameters**: ~530K trainable parameters

## Dataset

**CIFAR-10**: 60,000 color images (32×32 pixels) in 10 classes
- Training set: 50,000 images (5 batches)
- Test set: 10,000 images
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### Data Location

Place the CIFAR-10 binary dataset in:
```
cifar-10-batches-bin/
├── data_batch_1.bin
├── data_batch_2.bin
├── data_batch_3.bin
├── data_batch_4.bin
├── data_batch_5.bin
├── test_batch.bin
└── batches.meta.txt
```

Download from: https://www.cs.toronto.edu/~kriz/cifar.html (Binary version for C programs)

## Quick Start

### CPU Version
```bash
cd CPU
make
make run
```

### GPU Naive Version
```bash
cd GPU_naive
make
make run
```

### GPU Optimized Version (Phase 3)
```bash
cd GPU_optimized
make
make run
```

## Performance Comparison

| Implementation | Epoch Time | Speedup | Notes |
|---------------|------------|---------|-------|
| CPU Baseline | ~300-600s | 1× | Sequential C code |
| GPU Naive | ~30-60s | 5-20× | Direct CUDA translation |
| GPU Optimized | ~5-15s | 50-100× | cuDNN/cuBLAS + optimizations |

*Times are approximate and depend on hardware*

## Project Structure

```
Parallel-Programing-Capstone/
├── README.md                    # This file
├── CSC14120_2025_Final Project.pdf
├── cifar-10-batches-bin/        # CIFAR-10 dataset
│   ├── data_batch_*.bin
│   └── test_batch.bin
│
├── CPU/                         # Phase 1: CPU Baseline
│   ├── main.c
│   ├── cnn.{c,h}
│   ├── forward.{c,h}
│   ├── backward.{c,h}
│   ├── data_loader.{c,h}
│   ├── Makefile
│   └── README.md
│
├── GPU_naive/                   # Phase 2: Naive GPU
│   ├── main.cu
│   ├── cnn.{cu,cuh}
│   ├── forward.{cu,cuh}
│   ├── backward.{cu,cuh}
│   ├── data_loader.{c,h}
│   ├── Makefile
│   └── README.md
│
└── GPU_optimized/               # Phase 3: Optimized GPU
    └── (To be implemented)
```

## Requirements

### CPU Implementation
- GCC or compatible C compiler
- C99 standard support
- ~2GB RAM

### GPU Implementations
- NVIDIA GPU with CUDA support (Compute Capability 5.0+)
- CUDA Toolkit 10.0 or later
- NVCC compiler
- ~2GB GPU memory
- For optimized version: cuDNN library

## Training Configuration

Default hyperparameters (consistent across all implementations):
- **Batch size**: 64
- **Learning rate**: 0.001
- **Optimizer**: SGD (Stochastic Gradient Descent)
- **Epochs**: 10
- **Weight initialization**: He initialization
- **Loss function**: Cross-entropy

## Features

### All Implementations
- ✅ Forward propagation (Conv → ReLU → Pool → FC → Softmax)
- ✅ Backward propagation with gradient computation
- ✅ SGD weight updates
- ✅ Training loss and accuracy tracking
- ✅ Test set evaluation
- ✅ Progress bar during training
- ✅ Timing measurements

### GPU-Specific Features
- ✅ Device memory management
- ✅ Kernel launch optimizations
- ✅ Host-device data transfers
- ✅ CUDA error checking

## Development Phases

### ✅ Phase 1: CPU Baseline (Complete)
- Sequential implementation in C
- Establishes correctness baseline
- Reference for accuracy validation

### ✅ Phase 2: Naive GPU Implementation (Complete)
- Direct translation to CUDA kernels
- Basic parallelization
- Baseline GPU performance measurement

### ⏳ Phase 3: Optimized GPU Implementation (Pending)
- Shared memory usage
- Memory coalescing
- cuBLAS for matrix operations
- cuDNN for convolutions
- Kernel fusion
- Minimized CPU-GPU transfers

## Building from Source

### Prerequisites
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Check GCC
gcc --version
```

### Compilation
Each implementation has its own Makefile:
```bash
cd <implementation_folder>
make          # Build
make clean    # Clean build artifacts
make run      # Build and execute
make help     # Show available targets
```

## Expected Results

### Untrained Model (10 epochs)
- Training accuracy: ~25-35%
- Test accuracy: ~20-30%
- Random baseline: 10%

*Note: This is expected for only 10 training epochs. Full training would require 50-100+ epochs for 70-80% accuracy.*

### Sample Output
```
=== CIFAR-10 CNN Training ===
Batch size: 64
Learning rate: 0.0010
Number of epochs: 10

Loading training data...
Loaded 50000 training images
Loading test data...
Loaded 10000 test images

Model Architecture:
  Input: 32x32x3
  Conv1: 32 filters, 3x3 kernel -> 32x32x32
  ...

Epoch 1/10:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.1234, Acc: 0.2456
  Epoch completed in 45.32 seconds
Test Loss: 2.0123, Test Accuracy: 0.2789

...

=== Training Complete ===
Total training time: 453.20 seconds
Average time per epoch: 45.32 seconds
```

## Troubleshooting

### CPU Version
- **Compilation errors**: Ensure C99 support (`-std=c99`)
- **Slow performance**: Normal for CPU, expected ~5-10 minutes per epoch

### GPU Version
- **"No CUDA devices found"**: Check GPU drivers with `nvidia-smi`
- **Out of memory**: Reduce batch size in `main.cu`
- **Compilation errors**: Verify CUDA Toolkit installation and compute capability
- **Wrong results**: Ensure same random seed as CPU version

### Data Loading
- **Cannot open file**: Verify CIFAR-10 binary files in `../cifar-10-batches-bin/`
- **Incorrect path**: Adjust `data_dir` in `main.c`/`main.cu`

## Performance Optimization Tips

### CPU
- Use `-O3` optimization flag (already in Makefile)
- Consider OpenMP for multi-core parallelization (future work)

### GPU
- Adjust block/grid dimensions for your GPU architecture
- Update `-arch=sm_XX` in Makefile for your GPU generation
- Monitor GPU utilization with `nvidia-smi`

## Future Enhancements

- [ ] Data augmentation (random crops, flips)
- [ ] Learning rate scheduling
- [ ] Batch normalization
- [ ] Dropout regularization
- [ ] Adam optimizer
- [ ] Model checkpointing
- [ ] Multi-GPU training
- [ ] Mixed precision training (FP16)

## Project Goals

This project demonstrates:
1. **Baseline Performance**: CPU implementation for correctness
2. **Basic Parallelization**: Naive GPU speedup measurement
3. **Advanced Optimization**: Optimized GPU with library integration
4. **Performance Analysis**: Speedup comparisons and profiling
5. **Parallel Programming Concepts**: CUDA kernels, memory management, synchronization

## License

Academic project for CSC14120 Parallel Programming (2025)

## References

- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- CUDA Programming Guide: https://docs.nvidia.com/cuda/
- cuDNN Documentation: https://docs.nvidia.com/deeplearning/cudnn/
