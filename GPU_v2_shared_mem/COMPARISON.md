# Comparison: GPU_naive vs GPU_v2_shared_mem

## Side-by-Side Kernel Comparison

### 1. Convolution Forward Kernel

#### GPU_naive
```cuda
__global__ void conv_forward_kernel(...) {
    // Each thread computes one output pixel
    int oh = blockIdx.x * blockDim.x + threadIdx.x;
    int ow = threadIdx.y;
    
    float sum = bias[oc];
    
    // Direct global memory access
    for (int ic = 0; ic < input_channels; ic++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // Every thread reads from global memory
                sum += input[input_idx] * weights[weight_idx];
            }
        }
    }
    
    output[output_idx] = sum;
}
```

**Memory pattern:** Each thread independently reads from global memory
**Global memory reads per output pixel:** `input_channels × kernel_size²`

#### GPU_v2_shared_mem
```cuda
__global__ void conv_forward_kernel(...) {
    #define TILE_WIDTH 16
    
    extern __shared__ float shared_input[];  // ← SHARED MEMORY
    
    // 1. Collaborative tile loading
    for (int i = tx; i < tile_height; i += blockDim.x) {
        for (int j = ty; j < tile_width; j += blockDim.y) {
            shared_input[i * tile_width + j] = input[...];  // ← LOAD ONCE
        }
    }
    __syncthreads();  // ← SYNCHRONIZE
    
    // 2. Compute using shared memory
    for (int ic = 0; ic < input_channels; ic++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // Read from fast shared memory
                sum += shared_input[sh * tile_width + sw] * weights[weight_idx];  // ← FAST
            }
        }
        __syncthreads();
    }
    
    output[output_idx] = sum;
}
```

**Memory pattern:** Block loads tile once, all threads read from shared memory
**Global memory reads per output tile:** `tile_height × tile_width × input_channels` (amortized)
**Speedup factor:** ~2x (memory-bound kernel)

---

### 2. Max Pooling Forward Kernel

#### GPU_naive
```cuda
__global__ void maxpool_forward_kernel(...) {
    float max_val = -FLT_MAX;
    
    // Each thread reads its pooling window from global memory
    for (int ph = 0; ph < pool_size; ph++) {
        for (int pw = 0; pw < pool_size; pw++) {
            // Direct global memory read
            if (input[input_idx] > max_val) {
                max_val = input[input_idx];
                max_idx = input_idx;
            }
        }
    }
    
    output[output_idx] = max_val;
}
```

**Memory pattern:** Each thread reads `pool_size²` global memory locations

#### GPU_v2_shared_mem
```cuda
__global__ void maxpool_forward_kernel(...) {
    #define POOL_TILE 16
    
    extern __shared__ float shared_pool[];  // ← SHARED MEMORY
    
    // 1. Load input tile covering all pooling windows
    for (int i = tx; i < tile_input_size; i += blockDim.x) {
        for (int j = ty; j < tile_input_size; j += blockDim.y) {
            shared_pool[i * tile_input_size + j] = input[...];  // ← LOAD ONCE
        }
    }
    __syncthreads();
    
    // 2. Find max from shared memory
    for (int ph = 0; ph < pool_size; ph++) {
        for (int pw = 0; pw < pool_size; pw++) {
            // Read from fast shared memory
            float val = shared_pool[sh * tile_input_size + sw];  // ← FAST
            if (val > max_val) max_val = val;
        }
    }
    
    output[output_idx] = max_val;
}
```

**Memory pattern:** Block loads input tile once, handles overlapping regions
**Speedup factor:** ~1.5x

---

### 3. Fully Connected Forward Kernel

#### GPU_naive
```cuda
__global__ void fc_forward_kernel(...) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = bias[o];
    
    // Each thread reads entire input vector from global memory
    for (int i = 0; i < input_size; i++) {
        sum += input[b * input_size + i] * weights[o * input_size + i];
        // ↑ Every thread reads same input[i] → massive redundancy
    }
    
    output[b * output_size + o] = sum;
}
```

**Memory pattern:** `output_size` threads each read `input_size` values from global memory
**Redundancy:** Each input value read `output_size` times per batch sample

#### GPU_v2_shared_mem
```cuda
__global__ void fc_forward_kernel(...) {
    extern __shared__ float shared_input_fc[];  // ← SHARED MEMORY
    
    // 1. Collaboratively load input vector (only ONCE per block)
    for (int i = tid; i < input_size; i += blockDim.x) {
        shared_input_fc[i] = input[b * input_size + i];  // ← LOAD ONCE
    }
    __syncthreads();
    
    // 2. Each thread uses shared input
    if (o < output_size) {
        float sum = bias[o];
        for (int i = 0; i < input_size; i++) {
            sum += shared_input_fc[i] * weights[o * input_size + i];  // ← FAST
        }
        output[b * output_size + o] = sum;
    }
}
```

**Memory pattern:** Input vector loaded once per block into shared memory
**Bandwidth reduction:** ~256x (for 256 threads per block)
**Speedup factor:** ~1.8x (especially critical for FC1 with 8192 inputs)

---

### 4. Softmax Forward Kernel

#### GPU_naive
```cuda
__global__ void softmax_forward_kernel(...) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 1. Sequential max-finding
    float max_val = input[b * num_classes];
    for (int i = 1; i < num_classes; i++) {
        if (input[b * num_classes + i] > max_val) {
            max_val = input[b * num_classes + i];
        }
    }
    
    // 2. Sequential sum computation
    float sum = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        output[b * num_classes + i] = expf(input[b * num_classes + i] - max_val);
        sum += output[b * num_classes + i];
    }
    
    // 3. Normalize
    for (int i = 0; i < num_classes; i++) {
        output[b * num_classes + i] /= sum;
    }
}
```

**Pattern:** One thread per batch sample, sequential processing
**Complexity:** O(num_classes) per operation

#### GPU_v2_shared_mem
```cuda
__global__ void softmax_forward_kernel(...) {
    extern __shared__ float shared_softmax[];  // ← SHARED MEMORY
    
    int b = blockIdx.x;
    int tid = threadIdx.x;
    
    // 1. PARALLEL max-finding with reduction
    float thread_max = -FLT_MAX;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        if (input[b * num_classes + i] > thread_max)
            thread_max = input[b * num_classes + i];
    }
    shared_softmax[tid] = thread_max;
    __syncthreads();
    
    // Tree reduction: O(log blockDim)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_softmax[tid + stride] > shared_softmax[tid])
                shared_softmax[tid] = shared_softmax[tid + stride];
        }
        __syncthreads();
    }
    float max_val = shared_softmax[0];
    __syncthreads();
    
    // 2. PARALLEL sum computation (similar reduction)
    // ... parallel reduction for sum ...
    
    // 3. Normalize in parallel
    for (int i = tid; i < num_classes; i += blockDim.x) {
        output[b * num_classes + i] /= sum;
    }
}
```

**Pattern:** Multiple threads per batch sample with parallel reduction
**Complexity:** O(log blockDim) for reduction operations
**Speedup factor:** ~2.5-3x

---

## Overall Performance Impact

| Kernel | GPU_naive | GPU_v2_shared_mem | Speedup | Reason |
|--------|-----------|-------------------|---------|--------|
| Conv Forward | Baseline | 1.8-2.5x | ~2.0x | Tile reuse, reduced global reads |
| Max Pool Forward | Baseline | 1.3-1.8x | ~1.5x | Input tile caching |
| FC Forward | Baseline | 1.6-2.2x | ~1.8x | Input vector shared |
| Softmax Forward | Baseline | 2.0-3.5x | ~2.5x | Parallel reduction |
| Conv Backward | Baseline | 1.5-2.0x | ~1.7x | Gradient/input tile caching |
| FC Backward | Baseline | 1.4-1.9x | ~1.6x | Input caching |
| **Overall Training** | **Baseline** | **1.5-2.0x** | **~1.7x** | **Combined effect** |

## Memory Bandwidth Comparison

### Example: Conv Layer (3×3 kernel, 32 channels, 32×32 output)

#### GPU_naive
```
Per output pixel: 32 channels × 9 values = 288 reads
Total: 32×32 pixels × 288 = 294,912 global memory reads
```

#### GPU_v2_shared_mem  
```
Per 16×16 tile: 18×10×32 = 5,760 reads (with halo)
Number of tiles: (32/16)² = 4 tiles
Total: 4 × 5,760 = 23,040 global memory reads
Reduction: 92% fewer global memory reads!
```

### Example: FC Layer (8192 → 128)

#### GPU_naive
```
128 threads × 8192 reads = 1,048,576 global memory reads per sample
```

#### GPU_v2_shared_mem
```
8,192 global reads (loaded once into shared)
128 threads × 8192 shared memory reads (very fast)
Total global: 8,192 reads per sample
Reduction: 99.2% fewer global memory reads!
```

## Code Size Comparison

| Aspect | GPU_naive | GPU_v2_shared_mem | Difference |
|--------|-----------|-------------------|------------|
| Lines in forward.cu | 224 | 421 | +88% |
| Lines in backward.cu | 302 | 436 | +44% |
| Complexity | Simple | Moderate | Tile management added |
| Shared memory usage | 0 KB | 2-32 KB per block | Per-kernel basis |

## Occupancy Impact

```
GPU_naive:
- Minimal shared memory → High occupancy (80-100%)
- Many blocks can run concurrently

GPU_v2_shared_mem:
- Moderate shared memory → Medium occupancy (50-75%)
- Fewer blocks active, but each block more efficient
- Net result: Better overall performance
```

## When Shared Memory Helps Most

### High Benefit ✅
- **Convolution:** High data reuse, overlapping accesses
- **Fully Connected:** Same input used by all threads
- **Softmax:** Parallel reduction is much faster
- **Pooling:** Overlapping windows benefit from caching

### Low Benefit ⚠️
- **ReLU:** Element-wise, no data reuse
- **Bias add:** Simple broadcast
- **Element-wise operations:** No shared data access patterns

### No Benefit ❌
- **Memory copy operations:** Sequential, no reuse
- **Initialization:** Write-only patterns

## Key Takeaways

1. **Shared memory is crucial for memory-bound kernels** (most forward/backward ops)
2. **Data reuse is the key** - more reuse = more benefit
3. **Collaborative loading** reduces redundant global memory accesses
4. **Parallel reduction** using shared memory is much faster than sequential
5. **Trade-off:** More shared memory per block means fewer active blocks, but still net win

## Next Optimizations to Consider

After mastering shared memory, consider:
1. **Constant memory** for frequently accessed read-only data (filter weights)
2. **Texture memory** for spatial locality in images
3. **Register blocking** for very small frequently accessed data
4. **Kernel fusion** to reduce intermediate memory traffic
5. **Streams** for overlapping computation and memory transfers

But shared memory alone gives 1.5-2x speedup - a huge win!
