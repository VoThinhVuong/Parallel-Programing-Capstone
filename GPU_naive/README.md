# CIFAR-10 CNN - GPU Naive Implementation

This folder contains the **Phase 2: Naive GPU Implementation** of the CIFAR-10 CNN training project, as specified in the project requirements.

## Overview

This implementation ports the CPU baseline to CUDA with naive/direct translation of operations to GPU kernels. The goal is to establish a GPU baseline and measure initial speedup over the CPU version.

## Architecture

The CNN architecture is identical to the CPU version:

```
Input: 32x32x3
├─ Conv1: 32 filters, 3x3 kernel, stride 1, padding 1 → 32x32x32
├─ ReLU
├─ MaxPool1: 2x2, stride 2 → 16x16x32
├─ Conv2: 128 filters, 3x3 kernel, stride 1, padding 1 → 16x16x128
├─ ReLU
├─ MaxPool2: 2x2, stride 2 → 8x8x128
├─ FC1: 8192 → 128
├─ ReLU
├─ FC2: 128 → 10
└─ Softmax
```

## File Structure

```
GPU_naive/
├── cnn.cuh              # GPU data structures and declarations
├── cnn.cu               # Layer creation and weight initialization
├── forward.cuh          # Forward pass kernel declarations
├── forward.cu           # Forward pass CUDA kernels
├── backward.cuh         # Backward pass kernel declarations
├── backward.cu          # Backward pass CUDA kernels
├── data_loader.h        # Data loading interface (from CPU)
├── data_loader.c        # Data loading implementation (from CPU)
├── main.cu              # Main training loop
├── Makefile             # Build configuration
└── README.md            # This file
```

## Implementation Details

### Naive GPU Kernels

This implementation uses straightforward parallelization without optimizations:

1. **Convolution**: One thread per output element, direct nested loops
2. **ReLU**: One thread per element
3. **Max Pooling**: One thread per output element
4. **Fully Connected**: One thread per output neuron
5. **Softmax**: One thread per batch element
6. **Backward Pass**: Similar naive parallelization for gradient computation

### Memory Management

- All layer weights and activations stored on GPU (device memory)
- Batch data copied to GPU at start of each batch
- Output copied back to CPU for loss/accuracy calculation
- Gradients computed and stored entirely on GPU

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (Compute Capability 5.0 or higher)
- Minimum 2GB GPU memory recommended

### Software
- CUDA Toolkit (version 10.0 or later)
- NVCC compiler
- GCC or compatible C compiler (for data_loader.c)
- Windows or Linux OS

## Building

### Windows (PowerShell/CMD)

```bash
make
```

Or manually:
```bash
nvcc -O3 -arch=sm_50 -c main.cu -o main.o
nvcc -O3 -arch=sm_50 -c cnn.cu -o cnn.o
nvcc -O3 -arch=sm_50 -c forward.cu -o forward.o
nvcc -O3 -arch=sm_50 -c backward.cu -o backward.o
gcc -O3 -c data_loader.c -o data_loader.o
nvcc -O3 -arch=sm_50 main.o cnn.o forward.o backward.o data_loader.o -o cifar10_cnn_gpu.exe
```

### Linux

```bash
make
```

## Running

```bash
make run
```

Or directly:
```bash
./cifar10_cnn_gpu      # Linux
cifar10_cnn_gpu.exe    # Windows
```

**Note**: The CIFAR-10 dataset should be in `../cifar-10-batches-bin/` relative to the executable.

## Expected Output

```
=== CIFAR-10 CNN Training (GPU Naive Implementation) ===
Batch size: 64
Learning rate: 0.0010
Number of epochs: 10

Using GPU: NVIDIA GeForce RTX 3080
Compute Capability: 8.6
Global Memory: 10.00 GB

Loading training data...
Loading ../cifar-10-batches-bin/data_batch_1.bin...
...
Loaded 50000 training images

Loading test data...
Loading ../cifar-10-batches-bin/test_batch.bin...
Loaded 10000 test images

Creating CNN model on GPU...
Weights initialized and copied to GPU

Model Architecture:
  Input: 32x32x3
  Conv1: 32 filters, 3x3 kernel -> 32x32x32
  Pool1: 2x2 -> 16x16x32
  Conv2: 64 filters, 3x3 kernel -> 16x16x64
  Pool2: 2x2 -> 8x8x64
  FC1: 4096 -> 128
  FC2: 128 -> 10 (output)

Epoch 1/10:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.1234, Acc: 0.2456
  Epoch completed in 45.32 seconds - Avg Loss: 2.1234, Avg Acc: 0.2456
Evaluating on 10000 samples...
Test Loss: 2.0123, Test Accuracy: 0.2789
...
```

## Performance Metrics

The implementation tracks:
- **Per-epoch training time**: Total time for one complete pass through training data
- **Average loss and accuracy**: Computed across all batches
- **Test accuracy**: Evaluated after each epoch

## Comparison with CPU

To compare with CPU baseline:
1. Run CPU version: `cd ../CPU && make run`
2. Run GPU version: `cd ../GPU_naive && make run`
3. Compare epoch times and final accuracy

Expected speedup: **5-20x** depending on GPU hardware (naive implementation).

## Limitations (Naive Implementation)

This is an unoptimized baseline. Known inefficiencies:

1. **No shared memory usage** - All global memory accesses
2. **No memory coalescing optimization** - Irregular access patterns
3. **Atomic operations** in backward pass - Serialization bottleneck
4. **No kernel fusion** - Each operation separate kernel launch
5. **Host-device synchronization** - Frequent memory transfers for metrics
6. **No cuBLAS/cuDNN** - Hand-written kernels instead of optimized libraries

These will be addressed in Phase 3 (Optimized GPU Implementation).

## Debugging

### Check CUDA Installation
```bash
nvcc --version
nvidia-smi
```

### Common Issues

1. **"No CUDA devices found"**
   - Verify GPU is CUDA-capable: `nvidia-smi`
   - Check CUDA drivers installed

2. **Out of memory errors**
   - Reduce batch size in `main.cu`
   - Current default: 64

3. **Compilation errors**
   - Verify CUDA Toolkit installed
   - Check GPU compute capability and update `-arch=sm_XX` in Makefile

## Architecture Compatibility

The Makefile uses `-arch=sm_50` (Maxwell architecture, ~2014+). For newer GPUs:

- **Pascal (GTX 10xx)**: `-arch=sm_61`
- **Volta (Tesla V100)**: `-arch=sm_70`
- **Turing (RTX 20xx)**: `-arch=sm_75`
- **Ampere (RTX 30xx)**: `-arch=sm_80` or `-arch=sm_86`
- **Ada (RTX 40xx)**: `-arch=sm_89`

Update in Makefile for optimal performance on your GPU.

## Next Steps

Phase 3 will implement optimizations:
- Shared memory usage
- Memory coalescing
- cuBLAS for matrix operations
- cuDNN for convolutions
- Kernel fusion
- Reduced host-device transfers

## Project Context

This implementation satisfies **Phase 2** requirements:
- ✅ Direct translation of CPU code to CUDA kernels
- ✅ Naive parallelization without optimization
- ✅ Timing measurements for comparison
- ✅ Same accuracy as CPU version
- ✅ Establishes GPU baseline for optimization phase

## License

Part of CSC14120 Parallel Programming Final Project (2025).
