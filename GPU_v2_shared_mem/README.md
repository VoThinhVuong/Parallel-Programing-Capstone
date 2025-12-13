# GPU Shared Memory Optimized CNN Implementation

This directory contains a CUDA implementation of a CNN for CIFAR-10 classification optimized with **shared memory** techniques.

## Key Optimization: Shared Memory

This implementation focuses on **shared memory optimization only**, without other advanced techniques. Shared memory is on-chip memory that is much faster than global memory and is shared among threads in a block.

### Optimizations Applied

#### 1. **Convolution Forward Pass**
- **Shared Memory for Input Tiles**: Each thread block loads a tile of the input image (including halo region for padding) into shared memory
- **Benefits**: 
  - Reduces redundant global memory reads (multiple threads need the same input pixels)
  - Each input pixel is loaded once per block instead of multiple times per thread
  - Tile size: `(TILE_WIDTH + kernel_size - 1) x (blockDim.y + kernel_size - 1)`

#### 2. **Max Pooling Forward Pass**
- **Shared Memory for Input Tiles**: Loads pooling region tiles into shared memory
- **Benefits**:
  - Reduces global memory traffic when finding max values
  - Coalesced memory access patterns
  - Tile size accounts for stride and pool size: `TILE_WIDTH * stride + pool_size - stride`

#### 3. **Fully Connected Forward Pass**
- **Shared Memory for Input Vector**: Each block collaboratively loads input activations into shared memory
- **Benefits**:
  - All threads in a block can access the same input vector from fast shared memory
  - Reduces global memory bandwidth by `blockDim.x` times per batch sample
  - Size: `input_size * sizeof(float)` per block

#### 4. **Softmax Forward Pass**
- **Shared Memory for Parallel Reduction**: Uses shared memory for finding max and sum operations
- **Benefits**:
  - Efficient parallel reduction for max-finding (numerical stability)
  - Efficient parallel reduction for sum computation
  - Size: `blockDim.x * sizeof(float)` per block

#### 5. **Convolution Backward Pass**
- **Shared Memory for Output Gradient and Input Tiles**: Loads gradient and input tiles for efficient computation
- **Benefits**:
  - Reduces redundant memory accesses during gradient computation
  - Better memory access patterns for weight gradient accumulation

#### 6. **Fully Connected Backward Pass**
- **Shared Memory for Input Caching**: Uses shared memory for input values during gradient computation
- **Benefits**:
  - Reduces global memory reads when computing weight gradients
  - Improves memory bandwidth utilization

## Performance Characteristics

### Advantages of Shared Memory
1. **Lower Latency**: ~100x faster than global memory access
2. **Higher Bandwidth**: Much higher bandwidth than global memory per SM
3. **Data Reuse**: Enables efficient data sharing among threads in a block
4. **Reduced Global Memory Traffic**: Significant reduction in memory bandwidth requirements

### Memory Hierarchy
```
Global Memory (slowest, largest)
    ↓ Coalesced loads
L2 Cache
    ↓
L1 Cache / Shared Memory (fastest, smallest, per-SM)
    ↓ __syncthreads()
Thread Registers
```

## Architecture

The network architecture remains the same as GPU_naive:
- **Conv1**: 32 filters, 3×3 kernel, ReLU → 32×32×32
- **Pool1**: 2×2 max pooling → 16×16×32
- **Conv2**: 128 filters, 3×3 kernel, ReLU → 16×16×128
- **Pool2**: 2×2 max pooling → 8×8×128
- **FC1**: 8192 → 128, ReLU
- **FC2**: 128 → 10
- **Softmax**: 10 classes

## Build Instructions

### Prerequisites
- NVIDIA GPU with CUDA support (Compute Capability ≥ 5.0)
- CUDA Toolkit installed
- NVCC compiler
- Make utility

### Compilation

```bash
# Build everything
make

# Or build specific target
make cifar10_cnn_gpu_shared.exe     # Windows
make cifar10_cnn_gpu_shared          # Linux
```

### Running

```bash
# Training
./cifar10_cnn_gpu_shared.exe        # Windows
./cifar10_cnn_gpu_shared             # Linux

# Feature extraction (after training)
./extract_features_shared.exe        # Windows  
./extract_features_shared             # Linux
```

## Key Implementation Details

### Shared Memory Configuration

#### Convolution Kernel
```cuda
#define TILE_WIDTH 16
dim3 block(TILE_WIDTH, 8);  // 16x8 = 128 threads per block

// Shared memory size per block
int tile_height = TILE_WIDTH + kernel_size - 1;  // 16 + 3 - 1 = 18
int tile_width = 8 + kernel_size - 1;            // 8 + 3 - 1 = 10
int shared_mem_size = 18 * 10 * sizeof(float) = 720 bytes per input channel
```

#### Max Pooling Kernel
```cuda
#define POOL_TILE 16
dim3 block(POOL_TILE, 8);  // 16x8 = 128 threads per block

// Shared memory for input tile
int tile_input_size = POOL_TILE * stride + pool_size - stride;  // 16*2 + 2 - 2 = 32
int shared_mem_size = 32 * 32 * sizeof(float) = 4096 bytes per block
```

#### Fully Connected Kernel
```cuda
int threads = 256;
// Shared memory for input vector
int shared_mem_size = input_size * sizeof(float);  // e.g., 8192 * 4 = 32KB
```

#### Softmax Kernel
```cuda
int threads = 256;
// Shared memory for reduction
int shared_mem_size = threads * sizeof(float) = 1024 bytes per block
```

### Thread Synchronization

All kernels using shared memory employ `__syncthreads()` barriers to ensure:
1. All data is loaded into shared memory before computation
2. All threads complete computation before loading new data
3. Proper coordination between collaborative threads

### Memory Access Patterns

1. **Coalesced Global Memory Access**: Threads load consecutive memory locations
2. **Bank-Conflict-Free Shared Memory Access**: Carefully structured to minimize conflicts
3. **Register Pressure Management**: Balanced use of registers vs shared memory

## Comparison with GPU_naive

### Key Differences

| Aspect | GPU_naive | GPU_v2_shared_mem |
|--------|-----------|-------------------|
| Memory Access | Direct global memory for all data | Cached data in shared memory |
| Convolution | Each thread reads from global memory | Tile loaded once per block |
| Pooling | Direct global memory reads | Tile cached in shared memory |
| FC Forward | Each thread reads weights/input separately | Input cached in shared memory |
| Softmax | Sequential max/sum computation | Parallel reduction with shared memory |
| Memory Traffic | High global memory bandwidth usage | Reduced global memory traffic |

### Expected Performance Improvements

- **Convolution**: 1.5-2.5x speedup (depends on kernel size and tile reuse)
- **Max Pooling**: 1.3-1.8x speedup (benefit from input tile caching)
- **Fully Connected**: 1.5-2.0x speedup (shared input vector)
- **Softmax**: 2.0-3.0x speedup (parallel reduction)
- **Overall Training**: 1.5-2.0x speedup expected

## Files

- `forward.cu`: Forward pass kernels with shared memory optimization
- `backward.cu`: Backward pass kernels with shared memory optimization
- `cnn.cu`: CNN model structure and layer management
- `main.cu`: Training loop and evaluation
- `feature_extractor.cu`: Feature extraction for transfer learning
- `data_loader.cu`: CIFAR-10 data loading utilities
- `*.cuh` / `*.h`: Header files
- `Makefile`: Build configuration

## Shared Memory Limitations

### Hardware Constraints
- **Shared Memory Per Block**: Typically 48KB-96KB (GPU dependent)
- **Max Threads Per Block**: 1024
- **Impact**: Limits tile sizes and batch processing per block

### Our Usage
- Convolution: ~720 bytes per channel (well within limits)
- Pooling: 4KB per block (comfortable)
- FC layer: Up to 32KB for large layers (may need tuning for very large layers)
- Softmax: 1KB per block (minimal)

## Future Optimization Opportunities

This implementation focuses **only on shared memory**. Additional optimizations not yet applied:
- Constant memory for filter weights
- Texture memory for input features
- Tensor Cores for mixed-precision
- Stream/concurrent kernel execution
- Kernel fusion
- Register blocking

These will be explored in subsequent versions (GPU_v3, GPU_v4, etc.).

## Profiling Tips

To analyze shared memory usage:
```bash
# NVIDIA Nsight Compute
ncu --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared.sum ./cifar10_cnn_gpu_shared.exe

# Check shared memory bank conflicts
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum ./cifar10_cnn_gpu_shared.exe
```

## References

- CUDA C Programming Guide - Shared Memory
- NVIDIA CUDA Best Practices Guide
- "Optimizing Parallel Reduction in CUDA" by Mark Harris
- "Better Performance at Lower Occupancy" by Vasily Volkov

## Authors

Parallel Programming Capstone Project

## License

Educational use only.
