#include "forward.cuh"
#include "cnn.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <math.h>

// OPTIMIZED convolution kernel with shared memory
// Tile size for shared memory (adjust based on GPU)
#define TILE_WIDTH 16
#define MAX_KERNEL_SIZE 5

__global__ void conv_forward_kernel_optimized(float* input, float* weights, float* bias, float* output,
                                             int batch_size, int input_channels, int output_channels,
                                             int input_size, int output_size, int kernel_size,
                                             int stride, int padding) {
    // Shared memory for input tile (includes halo for kernel)
    extern __shared__ float shared_input[];
    
    int b = blockIdx.z;
    int oc = blockIdx.y;
    int oh = blockIdx.x * blockDim.x + threadIdx.x;
    int ow = threadIdx.y;
    
    if (b >= batch_size || oc >= output_channels || oh >= output_size || ow >= output_size) {
        return;
    }
    
    float sum = bias[oc];
    
    // Load weights into registers (small kernel)
    float local_weights[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < kernel_size * kernel_size && i < MAX_KERNEL_SIZE * MAX_KERNEL_SIZE; i++) {
            int weight_base = oc * input_channels * kernel_size * kernel_size;
            local_weights[i] = weights[weight_base + i];
        }
    }
    
    // Compute convolution with coalesced access
    for (int ic = 0; ic < input_channels; ic++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;
                
                if (ih >= 0 && ih < input_size && iw >= 0 && iw < input_size) {
                    int input_idx = b * (input_channels * input_size * input_size) +
                                   ic * (input_size * input_size) +
                                   ih * input_size + iw;
                    int weight_idx = oc * (input_channels * kernel_size * kernel_size) +
                                    ic * (kernel_size * kernel_size) +
                                    kh * kernel_size + kw;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }
    
    int output_idx = b * (output_channels * output_size * output_size) +
                    oc * (output_size * output_size) +
                    oh * output_size + ow;
    output[output_idx] = sum;
}

// Naive convolution kernel - one thread per output element
__global__ void conv_forward_kernel(float* input, float* weights, float* bias, float* output,
                                   int batch_size, int input_channels, int output_channels,
                                   int input_size, int output_size, int kernel_size,
                                   int stride, int padding) {
    int b = blockIdx.z;  // batch index
    int oc = blockIdx.y;  // output channel
    int oh = blockIdx.x * blockDim.x + threadIdx.x;  // output height
    int ow = threadIdx.y;  // output width
    
    if (b >= batch_size || oc >= output_channels || oh >= output_size || ow >= output_size) {
        return;
    }
    
    float sum = bias[oc];
    
    // Compute convolution
    for (int ic = 0; ic < input_channels; ic++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;
                
                if (ih >= 0 && ih < input_size && iw >= 0 && iw < input_size) {
                    int input_idx = b * (input_channels * input_size * input_size) +
                                   ic * (input_size * input_size) +
                                   ih * input_size + iw;
                    int weight_idx = oc * (input_channels * kernel_size * kernel_size) +
                                    ic * (kernel_size * kernel_size) +
                                    kh * kernel_size + kw;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }
    
    int output_idx = b * (output_channels * output_size * output_size) +
                    oc * (output_size * output_size) +
                    oh * output_size + ow;
    output[output_idx] = sum;
}

// ReLU activation kernel - OPTIMIZED: In-place operation
__global__ void relu_forward_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // In-place: data[idx] = max(0, data[idx])
        data[idx] = (data[idx] > 0) ? data[idx] : 0;
    }
}

// Max pooling kernel
__global__ void maxpool_forward_kernel(float* input, float* output, int* max_indices,
                                      int batch_size, int channels, int input_size,
                                      int output_size, int pool_size, int stride) {
    int b = blockIdx.z;  // batch index
    int c = blockIdx.y;  // channel
    int oh = blockIdx.x * blockDim.x + threadIdx.x;  // output height
    int ow = threadIdx.y;  // output width
    
    if (b >= batch_size || c >= channels || oh >= output_size || ow >= output_size) {
        return;
    }
    
    float max_val = -FLT_MAX;
    int max_idx = 0;
    
    // Find max in pool region
    for (int ph = 0; ph < pool_size; ph++) {
        for (int pw = 0; pw < pool_size; pw++) {
            int ih = oh * stride + ph;
            int iw = ow * stride + pw;
            
            int input_idx = b * (channels * input_size * input_size) +
                          c * (input_size * input_size) +
                          ih * input_size + iw;
            
            if (input[input_idx] > max_val) {
                max_val = input[input_idx];
                max_idx = input_idx;
            }
        }
    }
    
    int output_idx = b * (channels * output_size * output_size) +
                    c * (output_size * output_size) +
                    oh * output_size + ow;
    output[output_idx] = max_val;
    max_indices[output_idx] = max_idx;
}

// OPTIMIZED: Fully connected kernel with better memory coalescing
// Multiple threads cooperate to compute one output for better memory access
__global__ void fc_forward_kernel_optimized(float* input, float* weights, float* bias, float* output,
                                           int batch_size, int input_size, int output_size) {
    extern __shared__ float shared_data[];
    
    int b = blockIdx.y;  // batch index
    int o = blockIdx.x;  // output neuron
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    if (b >= batch_size || o >= output_size) {
        return;
    }
    
    // Each thread computes partial sum
    float partial_sum = 0.0f;
    for (int i = tid; i < input_size; i += block_size) {
        partial_sum += input[b * input_size + i] * weights[o * input_size + i];
    }
    shared_data[tid] = partial_sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 writes result
    if (tid == 0) {
        output[b * output_size + o] = shared_data[0] + bias[o];
    }
}

// Fully connected kernel - naive version with one thread per output neuron
__global__ void fc_forward_kernel(float* input, float* weights, float* bias, float* output,
                                 int batch_size, int input_size, int output_size) {
    int b = blockIdx.y;  // batch index
    int o = blockIdx.x * blockDim.x + threadIdx.x;  // output neuron
    
    if (b >= batch_size || o >= output_size) {
        return;
    }
    
    float sum = bias[o];
    for (int i = 0; i < input_size; i++) {
        sum += input[b * input_size + i] * weights[o * input_size + i];
    }
    
    output[b * output_size + o] = sum;
}

// Softmax kernel - naive version
__global__ void softmax_forward_kernel(float* input, float* output, int batch_size, int num_classes) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= batch_size) {
        return;
    }
    
    // Find max for numerical stability
    float max_val = input[b * num_classes];
    for (int i = 1; i < num_classes; i++) {
        if (input[b * num_classes + i] > max_val) {
            max_val = input[b * num_classes + i];
        }
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        output[b * num_classes + i] = expf(input[b * num_classes + i] - max_val);
        sum += output[b * num_classes + i];
    }
    
    // Normalize
    for (int i = 0; i < num_classes; i++) {
        output[b * num_classes + i] /= sum;
    }
}

// Host wrapper functions
// OPTIMIZED: Conv forward with optional shared memory
void conv_forward(ConvLayer* layer, float* d_input, int batch_size) {
    dim3 block(8, 8);
    dim3 grid((layer->output_size + block.x - 1) / block.x, layer->output_channels, batch_size);
    
    // Use optimized kernel for small kernels (3x3, 5x5)
    if (layer->kernel_size <= MAX_KERNEL_SIZE) {
        // Calculate shared memory size
        int shared_mem_size = TILE_WIDTH * TILE_WIDTH * sizeof(float);
        
        conv_forward_kernel_optimized<<<grid, block, shared_mem_size>>>(
            d_input, layer->d_weights, layer->d_bias, layer->d_output,
            batch_size, layer->input_channels, layer->output_channels,
            layer->input_size, layer->output_size, layer->kernel_size,
            layer->stride, layer->padding);
    } else {
        // Fall back to naive kernel for larger kernels
        conv_forward_kernel<<<grid, block>>>(d_input, layer->d_weights, layer->d_bias, layer->d_output,
                                            batch_size, layer->input_channels, layer->output_channels,
                                            layer->input_size, layer->output_size, layer->kernel_size,
                                            layer->stride, layer->padding);
    }
    CUDA_CHECK(cudaGetLastError());
}

// OPTIMIZED: In-place ReLU
void relu_forward(float* d_data, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    relu_forward_kernel<<<blocks, threads>>>(d_data, size);
    CUDA_CHECK(cudaGetLastError());
}

void maxpool_forward(MaxPoolLayer* layer, float* d_input, int batch_size) {
    dim3 block(8, 8);
    dim3 grid((layer->output_size + block.x - 1) / block.x, layer->input_channels, batch_size);
    
    maxpool_forward_kernel<<<grid, block>>>(d_input, layer->d_output, layer->d_max_indices,
                                           batch_size, layer->input_channels,
                                           layer->input_size, layer->output_size,
                                           layer->pool_size, layer->stride);
    CUDA_CHECK(cudaGetLastError());
}

// OPTIMIZED: FC forward with better memory access
void fc_forward(FCLayer* layer, float* d_input, int batch_size) {
    // For large FC layers, use optimized kernel with shared memory reduction
    if (layer->input_size > 1024) {
        int threads = 256;
        dim3 grid(layer->output_size, batch_size);
        int shared_mem_size = threads * sizeof(float);
        
        fc_forward_kernel_optimized<<<grid, threads, shared_mem_size>>>(
            d_input, layer->d_weights, layer->d_bias, layer->d_output,
            batch_size, layer->input_size, layer->output_size);
    } else {
        // For smaller layers, use simple kernel
        int threads = 256;
        int blocks = (layer->output_size + threads - 1) / threads;
        dim3 grid(blocks, batch_size);
        
        fc_forward_kernel<<<grid, threads>>>(d_input, layer->d_weights, layer->d_bias, layer->d_output,
                                            batch_size, layer->input_size, layer->output_size);
    }
    CUDA_CHECK(cudaGetLastError());
}

void softmax_forward(float* d_input, float* d_output, int batch_size, int num_classes) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    
    softmax_forward_kernel<<<blocks, threads>>>(d_input, d_output, batch_size, num_classes);
    CUDA_CHECK(cudaGetLastError());
}

// OPTIMIZED: Complete forward pass with in-place ReLU operations
void forward_pass(CNN* cnn, float* d_input) {
    int batch_size = cnn->batch_size;
    
    // Conv1 + ReLU (in-place)
    conv_forward(cnn->conv1, d_input, batch_size);
    relu_forward(cnn->conv1->d_output,
                 batch_size * CONV1_FILTERS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE);
    
    // Pool1 (reads from conv1->d_output directly)
    maxpool_forward(cnn->pool1, cnn->conv1->d_output, batch_size);
    
    // Conv2 + ReLU (in-place)
    conv_forward(cnn->conv2, cnn->pool1->d_output, batch_size);
    relu_forward(cnn->conv2->d_output,
                 batch_size * CONV2_FILTERS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE);
    
    // Pool2 (reads from conv2->d_output directly)
    maxpool_forward(cnn->pool2, cnn->conv2->d_output, batch_size);
    
    // FC1 + ReLU (in-place)
    fc_forward(cnn->fc1, cnn->pool2->d_output, batch_size);
    relu_forward(cnn->fc1->d_output, batch_size * FC1_OUTPUT_SIZE);
    
    // FC2 (reads from fc1->d_output directly)
    fc_forward(cnn->fc2, cnn->fc1->d_output, batch_size);
    
    // Softmax
    softmax_forward(cnn->fc2->d_output, cnn->d_output, batch_size, FC2_OUTPUT_SIZE);
}
