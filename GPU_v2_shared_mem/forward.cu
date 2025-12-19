#include "forward.cuh"
#include "cnn.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <math.h>

// Convolution kernel with shared memory for input tile
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
    
    // Shared memory for input tile (per input channel)
    // Size: (blockDim.x + kernel_size - 1) x (blockDim.y + kernel_size - 1)
    extern __shared__ float shared_input[];
    
    int tile_height = blockDim.x + kernel_size - 1;
    int tile_width = blockDim.y + kernel_size - 1;
    
    // Compute convolution
    for (int ic = 0; ic < input_channels; ic++) {
        // Collaboratively load input tile into shared memory
        int load_h = (tile_height + blockDim.x - 1) / blockDim.x;
        int load_w = (tile_width + blockDim.y - 1) / blockDim.y;
        
        for (int lh = 0; lh < load_h; lh++) {
            for (int lw = 0; lw < load_w; lw++) {
                int sh = threadIdx.x + lh * blockDim.x;
                int sw = threadIdx.y + lw * blockDim.y;
                
                if (sh < tile_height && sw < tile_width) {
                    int ih = blockIdx.x * blockDim.x + sh - padding;
                    int iw = sw - padding;
                    
                    if (ih >= 0 && ih < input_size && iw >= 0 && iw < input_size) {
                        int input_idx = b * (input_channels * input_size * input_size) +
                                       ic * (input_size * input_size) +
                                       ih * input_size + iw;
                        shared_input[sh * tile_width + sw] = input[input_idx];
                    } else {
                        shared_input[sh * tile_width + sw] = 0.0f;  // padding
                    }
                }
            }
        }
        
        __syncthreads();
        
        // Compute convolution using shared memory
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int sh = threadIdx.x + kh;
                int sw = threadIdx.y + kw;
                
                int weight_idx = oc * (input_channels * kernel_size * kernel_size) +
                                ic * (kernel_size * kernel_size) +
                                kh * kernel_size + kw;
                
                sum += shared_input[sh * tile_width + sw] * weights[weight_idx];
            }
        }
        
        __syncthreads();
    }
    
    int output_idx = b * (output_channels * output_size * output_size) +
                    oc * (output_size * output_size) +
                    oh * output_size + ow;
    output[output_idx] = sum;
}

// ReLU activation kernel (no shared memory optimization needed)
__global__ void relu_forward_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input[idx] > 0) ? input[idx] : 0;
    }
}

// Max pooling kernel with shared memory for input tile
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
    
    // Shared memory for input tile
    extern __shared__ float shared_pool[];
    
    int tile_height = blockDim.x * stride + pool_size - stride;
    int tile_width = blockDim.y * stride + pool_size - stride;
    
    // Collaboratively load input tile
    int load_h = (tile_height + blockDim.x - 1) / blockDim.x;
    int load_w = (tile_width + blockDim.y - 1) / blockDim.y;
    
    for (int lh = 0; lh < load_h; lh++) {
        for (int lw = 0; lw < load_w; lw++) {
            int sh = threadIdx.x + lh * blockDim.x;
            int sw = threadIdx.y + lw * blockDim.y;
            
            if (sh < tile_height && sw < tile_width) {
                int ih = blockIdx.x * blockDim.x * stride + sh;
                int iw = sw;
                
                if (ih < input_size && iw < input_size) {
                    int input_idx = b * (channels * input_size * input_size) +
                                   c * (input_size * input_size) +
                                   ih * input_size + iw;
                    shared_pool[sh * tile_width + sw] = input[input_idx];
                } else {
                    shared_pool[sh * tile_width + sw] = -FLT_MAX;
                }
            }
        }
    }
    
    __syncthreads();
    
    // Find max in pool region using shared memory
    float max_val = -FLT_MAX;
    int max_idx = 0;
    
    for (int ph = 0; ph < pool_size; ph++) {
        for (int pw = 0; pw < pool_size; pw++) {
            int sh = threadIdx.x * stride + ph;
            int sw = threadIdx.y * stride + pw;
            
            float val = shared_pool[sh * tile_width + sw];
            if (val > max_val) {
                max_val = val;
                // Compute global index for backprop
                int ih = oh * stride + ph;
                int iw = ow * stride + pw;
                max_idx = b * (channels * input_size * input_size) +
                         c * (input_size * input_size) +
                         ih * input_size + iw;
            }
        }
    }
    
    int output_idx = b * (channels * output_size * output_size) +
                    c * (output_size * output_size) +
                    oh * output_size + ow;
    output[output_idx] = max_val;
    max_indices[output_idx] = max_idx;
}

// Fully connected kernel with shared memory for weights and input
__global__ void fc_forward_kernel(float* input, float* weights, float* bias, float* output,
                                 int batch_size, int input_size, int output_size) {
    // Use shared memory for input vector
    extern __shared__ float shared_input_fc[];
    
    int b = blockIdx.y;  // batch index
    int o = blockIdx.x * blockDim.x + threadIdx.x;  // output neuron
    int tid = threadIdx.x;
    
    if (b >= batch_size) {
        return;
    }
    
    // Collaboratively load input into shared memory
    for (int i = tid; i < input_size; i += blockDim.x) {
        shared_input_fc[i] = input[b * input_size + i];
    }
    
    __syncthreads();
    
    // Compute dot product
    if (o < output_size) {
        float sum = bias[o];
        
        // Use shared input
        for (int i = 0; i < input_size; i++) {
            sum += shared_input_fc[i] * weights[o * input_size + i];
        }
        
        output[b * output_size + o] = sum;
    }
}

// Softmax kernel with shared memory for reduction
__global__ void softmax_forward_kernel(float* input, float* output, int batch_size, int num_classes) {
    extern __shared__ float shared_softmax[];
    
    int b = blockIdx.x;
    int tid = threadIdx.x;
    
    if (b >= batch_size) {
        return;
    }
    
    // Find max using parallel reduction in shared memory
    float thread_max = -FLT_MAX;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        if (input[b * num_classes + i] > thread_max) {
            thread_max = input[b * num_classes + i];
        }
    }
    shared_softmax[tid] = thread_max;
    
    __syncthreads();
    
    // Reduce to find global max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_softmax[tid + stride] > shared_softmax[tid]) {
                shared_softmax[tid] = shared_softmax[tid + stride];
            }
        }
        __syncthreads();
    }
    
    float max_val = shared_softmax[0];
    __syncthreads();
    
    // Compute exp and sum using shared memory
    float thread_sum = 0.0f;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float exp_val = expf(input[b * num_classes + i] - max_val);
        output[b * num_classes + i] = exp_val;
        thread_sum += exp_val;
    }
    shared_softmax[tid] = thread_sum;
    
    __syncthreads();
    
    // Reduce to find total sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_softmax[tid] += shared_softmax[tid + stride];
        }
        __syncthreads();
    }
    
    float sum = shared_softmax[0];
    __syncthreads();
    
    // Normalize
    for (int i = tid; i < num_classes; i += blockDim.x) {
        output[b * num_classes + i] /= sum;
    }
}

// Host wrapper functions
void conv_forward(ConvLayer* layer, float* d_input, int batch_size) {
    dim3 block(8, 8);
    dim3 grid((layer->output_size + block.x - 1) / block.x, layer->output_channels, batch_size);
    
    // Calculate shared memory size
    int tile_height = block.x + layer->kernel_size - 1;
    int tile_width = block.y + layer->kernel_size - 1;
    int shared_mem_size = tile_height * tile_width * sizeof(float);
    
    conv_forward_kernel<<<grid, block, shared_mem_size>>>(
        d_input, layer->d_weights, layer->d_bias, layer->d_output,
        batch_size, layer->input_channels, layer->output_channels,
        layer->input_size, layer->output_size, layer->kernel_size,
        layer->stride, layer->padding);
    CUDA_CHECK(cudaGetLastError());
}

void relu_forward(float* d_input, float* d_output, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    relu_forward_kernel<<<blocks, threads>>>(d_input, d_output, size);
    CUDA_CHECK(cudaGetLastError());
}

void maxpool_forward(MaxPoolLayer* layer, float* d_input, int batch_size) {
    dim3 block(8, 8);
    dim3 grid((layer->output_size + block.x - 1) / block.x, layer->input_channels, batch_size);
    
    // Calculate shared memory size
    int tile_height = block.x * layer->stride + layer->pool_size - layer->stride;
    int tile_width = block.y * layer->stride + layer->pool_size - layer->stride;
    int shared_mem_size = tile_height * tile_width * sizeof(float);
    
    maxpool_forward_kernel<<<grid, block, shared_mem_size>>>(
        d_input, layer->d_output, layer->d_max_indices,
        batch_size, layer->input_channels,
        layer->input_size, layer->output_size,
        layer->pool_size, layer->stride);
    CUDA_CHECK(cudaGetLastError());
}

void fc_forward(FCLayer* layer, float* d_input, int batch_size) {
    int threads = 256;
    int blocks = (layer->output_size + threads - 1) / threads;
    dim3 grid(blocks, batch_size);
    
    // Shared memory for input vector
    int shared_mem_size = layer->input_size * sizeof(float);
    
    fc_forward_kernel<<<grid, threads, shared_mem_size>>>(
        d_input, layer->d_weights, layer->d_bias, layer->d_output,
        batch_size, layer->input_size, layer->output_size);
    CUDA_CHECK(cudaGetLastError());
}

void softmax_forward(float* d_input, float* d_output, int batch_size, int num_classes) {
    int threads = 256;
    int blocks = batch_size;
    
    // Shared memory for reduction
    int shared_mem_size = threads * sizeof(float);
    
    softmax_forward_kernel<<<blocks, threads, shared_mem_size>>>(
        d_input, d_output, batch_size, num_classes);
    CUDA_CHECK(cudaGetLastError());
}

// Complete forward pass through the network
void forward_pass(CNN* cnn, float* d_input) {
    int batch_size = cnn->batch_size;
    
    // Conv1 + ReLU
    conv_forward(cnn->conv1, d_input, batch_size);
    relu_forward(cnn->conv1->d_output, cnn->d_conv1_relu,
                 batch_size * CONV1_FILTERS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE);
    
    // Pool1
    maxpool_forward(cnn->pool1, cnn->d_conv1_relu, batch_size);
    
    // Conv2 + ReLU
    conv_forward(cnn->conv2, cnn->pool1->d_output, batch_size);
    relu_forward(cnn->conv2->d_output, cnn->d_conv2_relu,
                 batch_size * CONV2_FILTERS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE);
    
    // Pool2
    maxpool_forward(cnn->pool2, cnn->d_conv2_relu, batch_size);
    
    // FC1 + ReLU
    fc_forward(cnn->fc1, cnn->pool2->d_output, batch_size);
    relu_forward(cnn->fc1->d_output, cnn->d_fc1_relu, batch_size * FC1_OUTPUT_SIZE);
    
    // FC2
    fc_forward(cnn->fc2, cnn->d_fc1_relu, batch_size);
    
    // Softmax
    softmax_forward(cnn->fc2->d_output, cnn->d_output, batch_size, FC2_OUTPUT_SIZE);
}
