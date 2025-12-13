# GPU_v2_shared_mem: Quick Reference

## What's New in This Version?

This implementation adds **shared memory optimization** to the GPU_naive baseline. This is the ONLY optimization technique applied - no other advanced features yet.

## Key Changes from GPU_naive

### Forward Pass (forward.cu)

#### 1. Convolution Kernel
**Before (naive):**
```cuda
// Each thread reads directly from global memory
sum += input[input_idx] * weights[weight_idx];
```

**After (shared memory):**
```cuda
// Load input tile into shared memory once per block
extern __shared__ float shared_input[];
// ... collaborative loading ...
__syncthreads();
// All threads read from fast shared memory
sum += shared_input[sh * tile_width + sw] * weights[weight_idx];
```

**Impact:** ~2x speedup expected

#### 2. Max Pooling Kernel
**Added:** Shared memory for pooling input tiles
**Impact:** ~1.5x speedup expected

#### 3. Fully Connected Kernel  
**Added:** Shared memory for input vector caching
**Impact:** ~1.8x speedup expected (critical for large FC layers)

#### 4. Softmax Kernel
**Added:** Parallel reduction using shared memory
**Impact:** ~2.5x speedup expected

### Backward Pass (backward.cu)

Similar optimizations applied to gradient computation kernels:
- Convolution backward: Shared memory for gradients and input tiles
- FC backward: Shared memory for input caching

## Files Modified

| File | Status | Changes |
|------|--------|---------|
| forward.cu | ✅ Rewritten | All kernels optimized with shared memory |
| backward.cu | ✅ Rewritten | Gradient kernels optimized |
| cnn.cu | ✔️ Copied | No changes (layer structure unchanged) |
| cnn.cuh | ✔️ Copied | No changes |
| forward.cuh | ✔️ Copied | No changes (same API) |
| backward.cuh | ✔️ Copied | No changes (same API) |
| main.cu | 🔧 Modified | Updated title to "Shared Memory Implementation" |
| data_loader.* | ✔️ Copied | No changes |
| feature_extractor.* | ✔️ Copied | No changes |
| Makefile | 🔧 Modified | Updated executable names |

## Build & Run

```bash
# Clean and build
make clean
make

# Run training
./cifar10_cnn_gpu_shared.exe    # Windows
./cifar10_cnn_gpu_shared         # Linux
```

## Expected Performance

Compared to GPU_naive:
- **Overall speedup:** 1.5-2.0x
- **Memory bandwidth:** Reduced by 40-60%
- **Shared memory usage:** 2-32KB per block (depending on kernel)

## Memory Usage Per Kernel

```
Conv:     ~5KB per block (input tile + halo)
Pooling:  ~4KB per block (input tile)
FC:       ~32KB per block (input vector for FC1, 512B for FC2)
Softmax:  ~1KB per block (reduction buffer)
```

## Profiling Commands

```bash
# Check shared memory usage
ncu --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared.sum ./cifar10_cnn_gpu_shared.exe

# Compare global memory reads
ncu --metrics lts__t_sectors_op_read.sum ./cifar10_cnn_gpu_shared.exe

# Check bank conflicts (should be low)
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum ./cifar10_cnn_gpu_shared.exe
```

## What's NOT Included Yet

These optimizations are planned for future versions:
- ❌ Constant memory for filters
- ❌ Texture memory
- ❌ Stream/concurrent execution  
- ❌ Kernel fusion
- ❌ Tensor Cores
- ❌ Mixed precision (FP16)
- ❌ Warp-level primitives

## Code Structure

```
GPU_v2_shared_mem/
├── forward.cu              ← Main changes: all kernels use shared memory
├── backward.cu             ← Main changes: gradient kernels optimized
├── main.cu                 ← Minor: updated title
├── Makefile                ← Minor: updated executable names
├── cnn.cu/cuh             ← Same as GPU_naive
├── data_loader.cu/h       ← Same as GPU_naive
├── feature_extractor.*    ← Same as GPU_naive
├── README.md              ← Comprehensive documentation
└── SHARED_MEMORY_OPTIMIZATION.md  ← Detailed technical explanation
```

## Documentation

1. **README.md** - Overview, build instructions, architecture
2. **SHARED_MEMORY_OPTIMIZATION.md** - Detailed technical explanation of each optimization
3. This file (QUICK_REFERENCE.md) - Quick summary of changes

## Testing

The implementation maintains identical algorithm correctness to GPU_naive:
- Same network architecture
- Same weight initialization  
- Same forward/backward computations
- Same training procedure
- Should produce identical accuracy results (within floating-point precision)

Only the **performance** should improve due to better memory access patterns.

## Next Steps

After verifying this implementation:
1. Profile and measure actual speedup vs GPU_naive
2. Analyze shared memory bank conflicts
3. Tune tile sizes for your specific GPU
4. Consider implementing GPU_v3 with additional optimizations

## Quick Test

```bash
# Build both versions
cd ../GPU_naive && make && cd ../GPU_v2_shared_mem && make

# Run both (short test)
# Compare execution times from output

# Or use profiler for detailed comparison
ncu --metrics dram__bytes_read.sum ../GPU_naive/cifar10_cnn_gpu.exe
ncu --metrics dram__bytes_read.sum ./cifar10_cnn_gpu_shared.exe
```

## Troubleshooting

### Build Errors
- Ensure CUDA Toolkit is installed
- Check `arch=sm_XX` in Makefile matches your GPU compute capability
- Verify path to CIFAR-10 data in main.cu

### Runtime Errors
- **Out of shared memory:** Reduce tile sizes (TILE_WIDTH, POOL_TILE)
- **Incorrect results:** Check __syncthreads() placement
- **Slow performance:** Profile to identify bottlenecks

### Common Issues

**Issue:** Compile error about shared memory size
**Fix:** Your GPU has limited shared memory. Reduce TILE_WIDTH from 16 to 8 in forward.cu

**Issue:** Slower than naive version  
**Check:** 
- Is shared memory actually being used? (check with profiler)
- Are there excessive bank conflicts? (check with profiler)
- Is occupancy very low? (may need to reduce shared memory usage)

## References

See README.md and SHARED_MEMORY_OPTIMIZATION.md for more details.
