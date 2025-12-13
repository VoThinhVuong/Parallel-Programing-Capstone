# GPU_v2_shared_mem Documentation Index

Welcome to the GPU Shared Memory Optimized CNN implementation! This directory contains a complete implementation with comprehensive documentation.

## 📚 Documentation Guide

### For Quick Start
1. **[README.md](README.md)** - Start here! Overview, build instructions, architecture
2. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick summary of what changed from GPU_naive

### For Understanding the Optimizations
3. **[SHARED_MEMORY_OPTIMIZATION.md](SHARED_MEMORY_OPTIMIZATION.md)** - Deep dive into each kernel optimization
4. **[COMPARISON.md](COMPARISON.md)** - Side-by-side code comparison with GPU_naive

## 📁 Source Files

### Core Implementation (Modified from GPU_naive)
- **forward.cu** ⭐ - Forward pass kernels with shared memory (MAIN CHANGES)
- **backward.cu** ⭐ - Backward pass kernels with shared memory (MAIN CHANGES)
- **main.cu** 🔧 - Training loop (minor title update)
- **Makefile** 🔧 - Build system (updated executable names)

### Supporting Files (Unchanged from GPU_naive)
- **cnn.cu/cuh** - CNN model structure
- **data_loader.cu/h** - CIFAR-10 data loading
- **feature_extractor.cu/cuh** - Feature extraction utilities
- **extract_features.cu** - Feature extraction program
- **forward.cuh** - Forward pass API declarations
- **backward.cuh** - Backward pass API declarations

## 🎯 What to Read Based on Your Goal

### "I just want to build and run it"
→ Read: **README.md** (Build Instructions section)

### "I want to understand what changed"
→ Read: **QUICK_REFERENCE.md** → **COMPARISON.md**

### "I want to understand shared memory optimization in detail"
→ Read: **README.md** → **SHARED_MEMORY_OPTIMIZATION.md**

### "I'm implementing my own optimizations"
→ Study: **forward.cu** and **backward.cu** source code
→ Reference: **SHARED_MEMORY_OPTIMIZATION.md** for patterns

### "I want to compare performance with naive version"
→ Read: **COMPARISON.md** (Performance Impact section)
→ Use: Profiling commands in **QUICK_REFERENCE.md**

## 🔑 Key Concepts Explained

| Concept | Where to Learn |
|---------|---------------|
| What is shared memory? | README.md - Memory Hierarchy section |
| Why use shared memory? | SHARED_MEMORY_OPTIMIZATION.md - Introduction |
| How convolution uses it | SHARED_MEMORY_OPTIMIZATION.md - Convolution section |
| How to tune tile sizes | SHARED_MEMORY_OPTIMIZATION.md - Memory Calculation sections |
| Expected speedup | COMPARISON.md - Overall Performance Impact |
| Memory bandwidth savings | COMPARISON.md - Memory Bandwidth Comparison |
| Profiling and verification | QUICK_REFERENCE.md - Profiling Commands |

## 📊 Quick Stats

```
Optimization Type:     Shared Memory Only
Files Modified:        2 core files (forward.cu, backward.cu)
Code Size:            +132 lines in forward.cu, +134 lines in backward.cu
Shared Memory Used:    2-32 KB per block (kernel-dependent)
Expected Speedup:      1.5-2.0x overall
Memory Traffic:        Reduced by 40-90% (kernel-dependent)
Algorithm Changes:     None (identical correctness)
```

## 🚀 Quick Start Commands

```bash
# Build
make clean && make

# Run training
./cifar10_cnn_gpu_shared.exe         # Windows
./cifar10_cnn_gpu_shared              # Linux

# Profile shared memory usage
ncu --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared.sum ./cifar10_cnn_gpu_shared.exe

# Compare with naive (if both built)
ncu --metrics lts__t_sectors_op_read.sum ../GPU_naive/cifar10_cnn_gpu.exe ./cifar10_cnn_gpu_shared.exe
```

## 🎓 Learning Path

### Beginner
1. Read README.md overview
2. Build and run the program
3. Skim COMPARISON.md to see the differences
4. Look at one kernel (e.g., FC forward) in forward.cu

### Intermediate
1. Read SHARED_MEMORY_OPTIMIZATION.md thoroughly
2. Study forward.cu implementation in detail
3. Understand parallel reduction pattern in softmax
4. Profile and verify shared memory usage

### Advanced
1. Analyze backward.cu gradient computation patterns
2. Experiment with different tile sizes
3. Profile bank conflicts and occupancy
4. Consider implementing further optimizations

## 🔍 Kernel-Specific Documentation

| Kernel | Documentation Section | Source Code |
|--------|----------------------|-------------|
| Convolution Forward | SHARED_MEMORY_OPTIMIZATION.md §1 | forward.cu:8-99 |
| Max Pooling Forward | SHARED_MEMORY_OPTIMIZATION.md §2 | forward.cu:106-174 |
| FC Forward | SHARED_MEMORY_OPTIMIZATION.md §3 | forward.cu:177-209 |
| Softmax Forward | SHARED_MEMORY_OPTIMIZATION.md §4 | forward.cu:212-268 |
| Convolution Backward | SHARED_MEMORY_OPTIMIZATION.md §5 | backward.cu:84-173 |
| FC Backward | SHARED_MEMORY_OPTIMIZATION.md §5 | backward.cu:25-74 |

## 📈 Performance Expectations

Based on memory access pattern analysis:

```
Kernel Performance (vs GPU_naive):
├─ Convolution:      1.8-2.5x faster  ⭐⭐⭐⭐⭐
├─ Max Pooling:      1.3-1.8x faster  ⭐⭐⭐⭐
├─ Fully Connected:  1.6-2.2x faster  ⭐⭐⭐⭐⭐
├─ Softmax:          2.0-3.5x faster  ⭐⭐⭐⭐⭐
├─ ReLU:             ~1.0x (no change) ⭐
└─ Overall Training: 1.5-2.0x faster  ⭐⭐⭐⭐

Memory Bandwidth (vs GPU_naive):
├─ Convolution:      -92% global reads  ⭐⭐⭐⭐⭐
├─ Fully Connected:  -99% global reads  ⭐⭐⭐⭐⭐
└─ Overall:          -40 to -60%        ⭐⭐⭐⭐
```

## 🐛 Troubleshooting

**Problem:** Build errors about shared memory size
**Solution:** Reduce TILE_WIDTH in forward.cu (line 13, line 106)
**Reference:** QUICK_REFERENCE.md - Troubleshooting section

**Problem:** Slower than GPU_naive
**Solution:** Profile to verify shared memory is being used
**Reference:** QUICK_REFERENCE.md - Profiling Commands

**Problem:** Incorrect results
**Solution:** Check __syncthreads() placement
**Reference:** SHARED_MEMORY_OPTIMIZATION.md - General Best Practices

## 🔄 Version History

- **GPU_naive** - Baseline GPU implementation, no optimizations
- **GPU_v2_shared_mem** - Added shared memory optimization (THIS VERSION)
- **Future versions** - Additional optimizations planned

## 📞 Getting Help

1. Check the troubleshooting section in QUICK_REFERENCE.md
2. Review the specific kernel documentation in SHARED_MEMORY_OPTIMIZATION.md
3. Compare your results with expected performance in COMPARISON.md
4. Profile using commands in QUICK_REFERENCE.md

## ✅ Verification Checklist

After building and running:
- [ ] Program compiles without errors
- [ ] Training runs and shows progress
- [ ] Shared memory is being used (check with profiler)
- [ ] Performance is better than GPU_naive (1.5-2x expected)
- [ ] Results are correct (accuracy should be similar to naive)

## 🎯 Next Steps After This Version

Once you understand shared memory optimization:
1. **Measure** actual speedup on your hardware
2. **Profile** to identify remaining bottlenecks
3. **Experiment** with tile sizes for your GPU
4. **Plan** next optimization (constant memory, streams, etc.)
5. **Implement** GPU_v3 with additional techniques

---

## Quick Navigation

- [📖 Main Documentation](README.md)
- [⚡ Quick Reference](QUICK_REFERENCE.md)  
- [🔬 Detailed Optimization Guide](SHARED_MEMORY_OPTIMIZATION.md)
- [📊 Code Comparison](COMPARISON.md)
- [💻 Source Code](forward.cu)

Happy coding! 🚀
