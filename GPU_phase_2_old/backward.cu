#include "backward.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Softmax + Cross-Entropy backward kernel (combined for efficiency)
__global__ void softmax_cross_entropy_backward_kernel(float* output, uint8_t* labels, float* gradient,
                                                     int batch_size, int num_classes) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (b >= batch_size || i >= num_classes) {
        return;
    }
    
    int idx = b * num_classes + i;
    gradient[idx] = output[idx];
    if (i == labels[b]) {
        gradient[idx] -= 1.0f;
    }
    gradient[idx] /= batch_size;  // Average over batch
}

// FC backward kernel - naive version
__global__ void fc_backward_kernel(float* input, float* weights, float* output_gradient,
                                  float* weight_gradients, float* bias_gradients, float* input_gradients,
                                  int batch_size, int input_size, int output_size) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (o >= output_size) {
        return;
    }
    
    // Compute bias gradient (sum over batch)
    float bias_grad = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        bias_grad += output_gradient[b * output_size + o];
    }
    atomicAdd(&bias_gradients[o], bias_grad);
    
    // Compute weight gradients and input gradients
    for (int i = 0; i < input_size; i++) {
        float weight_grad = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            float out_grad = output_gradient[b * output_size + o];
            weight_grad += out_grad * input[b * input_size + i];
            atomicAdd(&input_gradients[b * input_size + i], out_grad * weights[o * input_size + i]);
        }
        atomicAdd(&weight_gradients[o * input_size + i], weight_grad);
    }
}

// ReLU backward kernel
__global__ void relu_backward_kernel(float* input, float* output_gradient, float* input_gradient, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input_gradient[idx] = (input[idx] > 0) ? output_gradient[idx] : 0;
    }
}

// Max pooling backward kernel
__global__ void maxpool_backward_kernel(float* output_gradient, int* max_indices, float* input_gradients,
                                       int batch_size, int channels, int input_size, int output_size) {
    int b = blockIdx.z;
    int c = blockIdx.y;
    int oh = blockIdx.x * blockDim.x + threadIdx.x;
    int ow = threadIdx.y;
    
    if (b >= batch_size || c >= channels || oh >= output_size || ow >= output_size) {
        return;
    }
    
    int output_idx = b * (channels * output_size * output_size) +
                    c * (output_size * output_size) +
                    oh * output_size + ow;
    int max_idx = max_indices[output_idx];
    
    atomicAdd(&input_gradients[max_idx], output_gradient[output_idx]);
}

// OPTIMIZED: Convolution backward kernel with shared memory for reduction
__global__ void conv_backward_kernel_optimized(float* input, float* weights, float* output_gradient,
                                              float* weight_gradients, float* bias_gradients, float* input_gradients,
                                              int batch_size, int input_channels, int output_channels,
                                              int input_size, int output_size, int kernel_size,
                                              int stride, int padding) {
    extern __shared__ float shared_mem[];
    float* shared_bias_grad = shared_mem;
    
    int oc = blockIdx.y;
    int oh = blockIdx.x * blockDim.x + threadIdx.x;
    int ow = threadIdx.y;
    int tid = threadIdx.x * blockDim.y + threadIdx.y;
    int block_size = blockDim.x * blockDim.y;
    
    if (oc >= output_channels || oh >= output_size || ow >= output_size) {
        return;
    }
    
    // Initialize shared memory
    if (tid < block_size) {
        shared_bias_grad[tid] = 0.0f;
    }
    __syncthreads();
    
    // Compute bias gradient using shared memory reduction
    float local_bias_grad = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        int output_idx = b * (output_channels * output_size * output_size) +
                        oc * (output_size * output_size) +
                        oh * output_size + ow;
        local_bias_grad += output_gradient[output_idx];
    }
    shared_bias_grad[tid] = local_bias_grad;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride_red = block_size / 2; stride_red > 0; stride_red >>= 1) {
        if (tid < stride_red) {
            shared_bias_grad[tid] += shared_bias_grad[tid + stride_red];
        }
        __syncthreads();
    }
    
    // Only thread 0 updates bias gradient
    if (tid == 0) {
        atomicAdd(&bias_gradients[oc], shared_bias_grad[0]);
    }
    
    // Compute weight and input gradients with coalesced access
    for (int b = 0; b < batch_size; b++) {
        int output_idx = b * (output_channels * output_size * output_size) +
                        oc * (output_size * output_size) +
                        oh * output_size + ow;
        float out_grad = output_gradient[output_idx];
        
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
                        
                        atomicAdd(&weight_gradients[weight_idx], out_grad * input[input_idx]);
                        atomicAdd(&input_gradients[input_idx], out_grad * weights[weight_idx]);
                    }
                }
            }
        }
    }
}

// Convolution backward kernel - naive version (for large kernels)
__global__ void conv_backward_kernel(float* input, float* weights, float* output_gradient,
                                    float* weight_gradients, float* bias_gradients, float* input_gradients,
                                    int batch_size, int input_channels, int output_channels,
                                    int input_size, int output_size, int kernel_size,
                                    int stride, int padding) {
    int oc = blockIdx.y;
    int oh = blockIdx.x * blockDim.x + threadIdx.x;
    int ow = threadIdx.y;
    
    if (oc >= output_channels || oh >= output_size || ow >= output_size) {
        return;
    }
    
    // Compute bias gradient (sum over batch and spatial dimensions)
    float bias_grad = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        int output_idx = b * (output_channels * output_size * output_size) +
                        oc * (output_size * output_size) +
                        oh * output_size + ow;
        bias_grad += output_gradient[output_idx];
    }
    atomicAdd(&bias_gradients[oc], bias_grad);
    
    // Compute weight and input gradients
    for (int b = 0; b < batch_size; b++) {
        int output_idx = b * (output_channels * output_size * output_size) +
                        oc * (output_size * output_size) +
                        oh * output_size + ow;
        float out_grad = output_gradient[output_idx];
        
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
                        
                        atomicAdd(&weight_gradients[weight_idx], out_grad * input[input_idx]);
                        atomicAdd(&input_gradients[input_idx], out_grad * weights[weight_idx]);
                    }
                }
            }
        }
    }
}

// Weight update kernel
__global__ void update_weights_kernel(float* weights, float* gradients, float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}

// Clear gradient kernel
__global__ void clear_gradients_kernel(float* gradients, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gradients[idx] = 0.0f;
    }
}

// Host wrapper functions
void softmax_cross_entropy_backward(float* d_output, uint8_t* d_labels, float* d_gradient,
                                    int batch_size, int num_classes) {
    dim3 block(16, 16);
    dim3 grid((batch_size + block.x - 1) / block.x, (num_classes + block.y - 1) / block.y);
    
    softmax_cross_entropy_backward_kernel<<<grid, block>>>(d_output, d_labels, d_gradient,
                                                           batch_size, num_classes);
    CUDA_CHECK(cudaGetLastError());
}

void fc_backward(FCLayer* layer, float* d_input, float* d_output_gradient, int batch_size) {
    // Clear gradients first
    int weight_size = layer->output_size * layer->input_size;
    int threads = 256;
    
    clear_gradients_kernel<<<(weight_size + threads - 1) / threads, threads>>>(layer->d_weight_gradients, weight_size);
    clear_gradients_kernel<<<(layer->output_size + threads - 1) / threads, threads>>>(layer->d_bias_gradients, layer->output_size);
    clear_gradients_kernel<<<(batch_size * layer->input_size + threads - 1) / threads, threads>>>(layer->d_input_gradients, batch_size * layer->input_size);
    
    // Compute gradients
    int blocks = (layer->output_size + threads - 1) / threads;
    fc_backward_kernel<<<blocks, threads>>>(d_input, layer->d_weights, d_output_gradient,
                                           layer->d_weight_gradients, layer->d_bias_gradients,
                                           layer->d_input_gradients,
                                           batch_size, layer->input_size, layer->output_size);
    CUDA_CHECK(cudaGetLastError());
}

void relu_backward(float* d_input, float* d_output_gradient, float* d_input_gradient, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    relu_backward_kernel<<<blocks, threads>>>(d_input, d_output_gradient, d_input_gradient, size);
    CUDA_CHECK(cudaGetLastError());
}

void maxpool_backward(MaxPoolLayer* layer, float* d_output_gradient, int batch_size) {
    int input_total_size = batch_size * layer->input_channels * layer->input_size * layer->input_size;
    int threads = 256;
    clear_gradients_kernel<<<(input_total_size + threads - 1) / threads, threads>>>(layer->d_input_gradients, input_total_size);
    
    dim3 block(8, 8);
    dim3 grid((layer->output_size + block.x - 1) / block.x, layer->input_channels, batch_size);
    
    maxpool_backward_kernel<<<grid, block>>>(d_output_gradient, layer->d_max_indices, layer->d_input_gradients,
                                            batch_size, layer->input_channels,
                                            layer->input_size, layer->output_size);
    CUDA_CHECK(cudaGetLastError());
}

// OPTIMIZED: Conv backward with shared memory
void conv_backward(ConvLayer* layer, float* d_input, float* d_output_gradient, int batch_size) {
    // Clear gradients
    int weight_size = layer->output_channels * layer->input_channels * layer->kernel_size * layer->kernel_size;
    int input_grad_size = batch_size * layer->input_channels * layer->input_size * layer->input_size;
    int threads = 256;
    
    clear_gradients_kernel<<<(weight_size + threads - 1) / threads, threads>>>(layer->d_weight_gradients, weight_size);
    clear_gradients_kernel<<<(layer->output_channels + threads - 1) / threads, threads>>>(layer->d_bias_gradients, layer->output_channels);
    clear_gradients_kernel<<<(input_grad_size + threads - 1) / threads, threads>>>(layer->d_input_gradients, input_grad_size);
    
    // Compute gradients with optimized kernel
    dim3 block(8, 8);
    dim3 grid((layer->output_size + block.x - 1) / block.x, layer->output_channels);
    
    if (layer->kernel_size <= 5) {
        // Use optimized kernel with shared memory for small kernels
        int shared_mem_size = block.x * block.y * sizeof(float);
        conv_backward_kernel_optimized<<<grid, block, shared_mem_size>>>(
            d_input, layer->d_weights, d_output_gradient,
            layer->d_weight_gradients, layer->d_bias_gradients,
            layer->d_input_gradients,
            batch_size, layer->input_channels, layer->output_channels,
            layer->input_size, layer->output_size, layer->kernel_size,
            layer->stride, layer->padding);
    } else {
        // Use naive kernel for large kernels
        conv_backward_kernel<<<grid, block>>>(d_input, layer->d_weights, d_output_gradient,
                                             layer->d_weight_gradients, layer->d_bias_gradients,
                                             layer->d_input_gradients,
                                             batch_size, layer->input_channels, layer->output_channels,
                                             layer->input_size, layer->output_size, layer->kernel_size,
                                             layer->stride, layer->padding);
    }
    CUDA_CHECK(cudaGetLastError());
}

// OPTIMIZED: Complete backward pass using shared gradient buffer
void backward_pass(CNN* cnn, float* d_input, uint8_t* d_labels) {
    int batch_size = cnn->batch_size;
    
    // Use shared gradient buffer (already allocated in CNN structure)
    // This buffer is large enough for all intermediate gradients
    float* d_shared_grad = cnn->d_shared_grad_buffer;
    
    // Softmax + Cross-Entropy gradient (writes to shared buffer)
    // Size: batch_size * 10 (small, fits in shared buffer)
    softmax_cross_entropy_backward(cnn->d_output, d_labels, d_shared_grad, batch_size, FC2_OUTPUT_SIZE);
    
    // FC2 backward (reads from shared buffer)
    fc_backward(cnn->fc2, cnn->fc1->d_output, d_shared_grad, batch_size);
    
    // ReLU backward (FC1) - use shared buffer for temporary gradient
    // Size: batch_size * 128 (fits in shared buffer)
    relu_backward(cnn->fc1->d_output, cnn->fc2->d_input_gradients, d_shared_grad,
                  batch_size * FC1_OUTPUT_SIZE);
    
    // FC1 backward (reads from shared buffer)
    fc_backward(cnn->fc1, cnn->pool2->d_output, d_shared_grad, batch_size);
    
    // Pool2 backward
    maxpool_backward(cnn->pool2, cnn->fc1->d_input_gradients, batch_size);
    
    // ReLU backward (Conv2) - reuse shared buffer
    // Size: batch_size * 128 * 16 * 16 = batch_size * 32768 (fits in shared buffer)
    relu_backward(cnn->conv2->d_output, cnn->pool2->d_input_gradients, d_shared_grad,
                  batch_size * CONV2_FILTERS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE);
    
    // Conv2 backward (reads from shared buffer)
    conv_backward(cnn->conv2, cnn->pool1->d_output, d_shared_grad, batch_size);
    
    // Pool1 backward
    maxpool_backward(cnn->pool1, cnn->conv2->d_input_gradients, batch_size);
    
    // ReLU backward (Conv1) - reuse shared buffer
    // Size: batch_size * 32 * 32 * 32 = batch_size * 32768 (fits in shared buffer)
    relu_backward(cnn->conv1->d_output, cnn->pool1->d_input_gradients, d_shared_grad,
                  batch_size * CONV1_FILTERS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE);
    
    // Conv1 backward (reads from shared buffer)
    conv_backward(cnn->conv1, d_input, d_shared_grad, batch_size);
    
    // No need to free - shared buffer is reused across iterations
}

// Update all weights
void update_weights(CNN* cnn, float learning_rate) {
    int threads = 256;
    
    // Update Conv1
    int conv1_weight_size = CONV1_FILTERS * INPUT_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE;
    update_weights_kernel<<<(conv1_weight_size + threads - 1) / threads, threads>>>(
        cnn->conv1->d_weights, cnn->conv1->d_weight_gradients, learning_rate, conv1_weight_size);
    update_weights_kernel<<<(CONV1_FILTERS + threads - 1) / threads, threads>>>(
        cnn->conv1->d_bias, cnn->conv1->d_bias_gradients, learning_rate, CONV1_FILTERS);
    
    // Update Conv2
    int conv2_weight_size = CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE;
    update_weights_kernel<<<(conv2_weight_size + threads - 1) / threads, threads>>>(
        cnn->conv2->d_weights, cnn->conv2->d_weight_gradients, learning_rate, conv2_weight_size);
    update_weights_kernel<<<(CONV2_FILTERS + threads - 1) / threads, threads>>>(
        cnn->conv2->d_bias, cnn->conv2->d_bias_gradients, learning_rate, CONV2_FILTERS);
    
    // Update FC1
    int fc1_weight_size = FC1_OUTPUT_SIZE * FC1_INPUT_SIZE;
    update_weights_kernel<<<(fc1_weight_size + threads - 1) / threads, threads>>>(
        cnn->fc1->d_weights, cnn->fc1->d_weight_gradients, learning_rate, fc1_weight_size);
    update_weights_kernel<<<(FC1_OUTPUT_SIZE + threads - 1) / threads, threads>>>(
        cnn->fc1->d_bias, cnn->fc1->d_bias_gradients, learning_rate, FC1_OUTPUT_SIZE);
    
    // Update FC2
    int fc2_weight_size = FC2_OUTPUT_SIZE * FC2_INPUT_SIZE;
    update_weights_kernel<<<(fc2_weight_size + threads - 1) / threads, threads>>>(
        cnn->fc2->d_weights, cnn->fc2->d_weight_gradients, learning_rate, fc2_weight_size);
    update_weights_kernel<<<(FC2_OUTPUT_SIZE + threads - 1) / threads, threads>>>(
        cnn->fc2->d_bias, cnn->fc2->d_bias_gradients, learning_rate, FC2_OUTPUT_SIZE);
    
    CUDA_CHECK(cudaGetLastError());
}
