# CIFAR-10 CNN - CPU Baseline Implementation

This directory contains the CPU baseline implementation for CIFAR-10 image classification using a Convolutional Neural Network (CNN).

## Project Structure

```
CPU/
├── main.c           - Main training loop and program entry point
├── data_loader.c/h  - CIFAR-10 binary data loading utilities
├── cnn.c/h          - CNN model structure and initialization
├── forward.c/h      - Forward propagation implementation
├── backward.c/h     - Backward propagation and weight updates
├── Makefile         - Build system
└── README.md        - This file
```

## Model Architecture

The CNN consists of the following layers:

1. **Input**: 32×32×3 (RGB images)
2. **Conv1**: 32 filters, 3×3 kernel, stride 1, padding 1 → 32×32×32
3. **ReLU**: Activation function
4. **Pool1**: 2×2 max pooling, stride 2 → 16×16×32
5. **Conv2**: 64 filters, 3×3 kernel, stride 1, padding 1 → 16×16×64
6. **ReLU**: Activation function
7. **Pool2**: 2×2 max pooling, stride 2 → 8×8×64
8. **FC1**: Fully connected, 4096 → 128
9. **ReLU**: Activation function
10. **FC2**: Fully connected, 128 → 10 (output)
11. **Softmax**: Output layer for classification

## Requirements

- GCC compiler (or compatible C compiler)
- Make build tool
- CIFAR-10 binary dataset (should be in `../cifar-10-batches-bin/`)

## Building

To build the project:

```bash
make
```

This will compile all source files and create the executable `cifar10_cnn`.

## Running

To run the training:

```bash
make run
```

Or directly:

```bash
./cifar10_cnn
```

## Configuration

Training parameters can be modified in `main.c`:

- `batch_size`: Number of samples per batch (default: 64)
- `num_epochs`: Number of training epochs (default: 10)
- `learning_rate`: Learning rate for SGD (default: 0.001)
- `data_dir`: Path to CIFAR-10 dataset (default: "../cifar-10-batches-bin")

## Data Format

The program expects CIFAR-10 data in binary format:
- Training: `data_batch_1.bin` through `data_batch_5.bin`
- Testing: `test_batch.bin`

Each file contains 10,000 images with the format:
- 1 byte: label (0-9)
- 3072 bytes: image data (32×32×3, in R, G, B channel order)

## Features

- **Data Loading**: Efficient binary file reading and normalization
- **Forward Propagation**: Convolution, pooling, fully connected layers
- **Backward Propagation**: Gradient computation via backpropagation
- **Training**: Stochastic Gradient Descent (SGD) optimizer
- **Evaluation**: Loss and accuracy metrics
- **Timing**: Performance measurements for benchmarking

## Performance Metrics

The program reports:
- Training loss and accuracy per epoch
- Test loss and accuracy after each epoch
- Total training time
- Average time per epoch

## Cleaning

To remove compiled files:

```bash
make clean
```

## Implementation Details

### Weight Initialization
- Uses He initialization for better convergence
- Random values scaled by √(2/n) where n is the number of input units

### Loss Function
- Cross-entropy loss for multi-class classification
- Combined with softmax for numerical stability

### Optimization
- Stochastic Gradient Descent (SGD)
- Fixed learning rate (no decay in baseline)

## Notes

- This is a CPU-only baseline implementation for comparison
- No parallelization or GPU acceleration is used in this version
- Memory is allocated upfront for efficiency
- All computations use single-precision floating-point (float)

## Future Enhancements (Next Phases)

This baseline will be extended with:
- OpenMP parallelization for CPU multi-threading
- CUDA implementation for GPU acceleration
- Optimized memory management
- Advanced optimization techniques (momentum, Adam, etc.)
