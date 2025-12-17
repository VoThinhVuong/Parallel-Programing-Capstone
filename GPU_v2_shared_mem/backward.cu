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

// FC backward kernel - naive version for correctness
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

// ReLU backward kernel (no shared memory optimization needed)
__global__ void relu_backward_kernel(float* input, float* output_gradient, float* input_gradient, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input_gradient[idx] = (input[idx] > 0) ? output_gradient[idx] : 0;
    }
}

// Max pooling backward kernel (no significant benefit from shared memory due to atomic scatter)
__global__ void maxpool_backward_kernel(float* output_gradient, int* max_indices, float* input_gradients,
                                       int batch_size, int channels, int input_size, int output_size) {
    int b = blockIdx.z;
    int c = blockIdx.y;

    int num_w_blocks = (output_size + blockDim.y - 1) / blockDim.y;
    int oh_block = blockIdx.x / num_w_blocks;
    int ow_block = blockIdx.x % num_w_blocks;

    int oh = oh_block * blockDim.x + threadIdx.x;
    int ow = ow_block * blockDim.y + threadIdx.y;
    
    if (b >= batch_size || c >= channels || oh >= output_size || ow >= output_size) {
        return;
    }
    
    int output_idx = b * (channels * output_size * output_size) +
                    c * (output_size * output_size) +
                    oh * output_size + ow;
    int max_idx = max_indices[output_idx];
    
    atomicAdd(&input_gradients[max_idx], output_gradient[output_idx]);
}

// Convolution backward kernel - using naive approach for correctness
__global__ void conv_backward_kernel(float* input, float* weights, float* output_gradient,
                                    float* weight_gradients, float* bias_gradients, float* input_gradients,
                                    int batch_size, int input_channels, int output_channels,
                                    int input_size, int output_size, int kernel_size,
                                    int stride, int padding) {
    int oc = blockIdx.y;

    int num_w_blocks = (output_size + blockDim.y - 1) / blockDim.y;
    int oh_block = blockIdx.x / num_w_blocks;
    int ow_block = blockIdx.x % num_w_blocks;

    int oh = oh_block * blockDim.x + threadIdx.x;
    int ow = ow_block * blockDim.y + threadIdx.y;
    
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

// Weight update kernel (no shared memory benefit)
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
    
    fc_backward_kernel<<<blocks, threads>>>(
        d_input, layer->d_weights, d_output_gradient,
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
    int grid_h = (layer->output_size + block.x - 1) / block.x;
    int grid_w = (layer->output_size + block.y - 1) / block.y;
    dim3 grid(grid_h * grid_w, layer->input_channels, batch_size);
    
    maxpool_backward_kernel<<<grid, block>>>(d_output_gradient, layer->d_max_indices, layer->d_input_gradients,
                                            batch_size, layer->input_channels,
                                            layer->input_size, layer->output_size);
    CUDA_CHECK(cudaGetLastError());
}

void conv_backward(ConvLayer* layer, float* d_input, float* d_output_gradient, int batch_size) {
    // Clear gradients
    int weight_size = layer->output_channels * layer->input_channels * layer->kernel_size * layer->kernel_size;
    int input_grad_size = batch_size * layer->input_channels * layer->input_size * layer->input_size;
    int threads = 256;
    
    clear_gradients_kernel<<<(weight_size + threads - 1) / threads, threads>>>(layer->d_weight_gradients, weight_size);
    clear_gradients_kernel<<<(layer->output_channels + threads - 1) / threads, threads>>>(layer->d_bias_gradients, layer->output_channels);
    clear_gradients_kernel<<<(input_grad_size + threads - 1) / threads, threads>>>(layer->d_input_gradients, input_grad_size);
    
    // Compute gradients
    dim3 block(8, 8);
    int grid_h = (layer->output_size + block.x - 1) / block.x;
    int grid_w = (layer->output_size + block.y - 1) / block.y;
    dim3 grid(grid_h * grid_w, layer->output_channels);
    
    conv_backward_kernel<<<grid, block>>>(
        d_input, layer->d_weights, d_output_gradient,
        layer->d_weight_gradients, layer->d_bias_gradients,
        layer->d_input_gradients,
        batch_size, layer->input_channels, layer->output_channels,
        layer->input_size, layer->output_size, layer->kernel_size,
        layer->stride, layer->padding);
    CUDA_CHECK(cudaGetLastError());
}

// Complete backward pass
void backward_pass(CNN* cnn, float* d_input, uint8_t* d_labels) {
    int batch_size = cnn->batch_size;
    
    // Allocate temporary gradient buffer
    float* d_fc2_grad;
    CUDA_CHECK(cudaMalloc(&d_fc2_grad, batch_size * FC2_OUTPUT_SIZE * sizeof(float)));
    
    // Softmax + Cross-Entropy gradient
    softmax_cross_entropy_backward(cnn->d_output, d_labels, d_fc2_grad, batch_size, FC2_OUTPUT_SIZE);
    
    // FC2 backward
    fc_backward(cnn->fc2, cnn->d_fc1_relu, d_fc2_grad, batch_size);
    
    // ReLU backward (FC1)
    relu_backward(cnn->fc1->d_output, cnn->fc2->d_input_gradients, cnn->d_fc1_relu_grad,
                  batch_size * FC1_OUTPUT_SIZE);
    
    // FC1 backward
    fc_backward(cnn->fc1, cnn->pool2->d_output, cnn->d_fc1_relu_grad, batch_size);
    
    // Pool2 backward
    maxpool_backward(cnn->pool2, cnn->fc1->d_input_gradients, batch_size);
    
    // ReLU backward (Conv2)
    relu_backward(cnn->conv2->d_output, cnn->pool2->d_input_gradients, cnn->d_conv2_relu_grad,
                  batch_size * CONV2_FILTERS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE);
    
    // Conv2 backward
    conv_backward(cnn->conv2, cnn->pool1->d_output, cnn->d_conv2_relu_grad, batch_size);
    
    // Pool1 backward
    maxpool_backward(cnn->pool1, cnn->conv2->d_input_gradients, batch_size);
    
    // ReLU backward (Conv1)
    relu_backward(cnn->conv1->d_output, cnn->pool1->d_input_gradients, cnn->d_conv1_relu_grad,
                  batch_size * CONV1_FILTERS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE);
    
    // Conv1 backward
    conv_backward(cnn->conv1, d_input, cnn->d_conv1_relu_grad, batch_size);
    
    CUDA_CHECK(cudaFree(d_fc2_grad));
}

// Update all weights
void update_weights(CNN* cnn, float learning_rate) {
    update_encoder_weights(cnn, learning_rate);
    update_classifier_weights(cnn, learning_rate);
}

// Update classifier weights only (FC1, FC2)
void update_classifier_weights(CNN* cnn, float learning_rate) {
    int threads = 256;

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

// Update encoder weights only (Conv1, Conv2)
void update_encoder_weights(CNN* cnn, float learning_rate) {
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

    CUDA_CHECK(cudaGetLastError());
}

// Gradient accumulation kernel
__global__ void accumulate_gradients_kernel(float* gradients, float* additional_gradients, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gradients[idx] += additional_gradients[idx];
    }
}

// Backpropagate reconstruction gradients to encoder (accumulates with classification gradients)
void backprop_reconstruction_to_encoder(CNN* cnn, float* d_input, float* d_pool2_gradient, int batch_size) {
    // Backward through Pool2 (accumulate gradients)
    maxpool_backward(cnn->pool2, d_pool2_gradient, batch_size);

    // Backward through ReLU for Conv2 (accumulate)
    int conv2_size = batch_size * CONV2_FILTERS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE;
    float* d_temp_grad;
    CUDA_CHECK(cudaMalloc(&d_temp_grad, conv2_size * sizeof(float)));
    relu_backward(cnn->conv2->d_output, cnn->pool2->d_input_gradients, d_temp_grad, conv2_size);

    int threads = 256;
    int blocks = (conv2_size + threads - 1) / threads;
    accumulate_gradients_kernel<<<blocks, threads>>>(cnn->d_conv2_relu_grad, d_temp_grad, conv2_size);
    CUDA_CHECK(cudaFree(d_temp_grad));

    // Backward through Conv2 (accumulate gradients)
    conv_backward(cnn->conv2, cnn->pool1->d_output, cnn->d_conv2_relu_grad, batch_size);

    // Backward through Pool1 (accumulate)
    maxpool_backward(cnn->pool1, cnn->conv2->d_input_gradients, batch_size);

    // Backward through ReLU for Conv1 (accumulate)
    int conv1_size = batch_size * CONV1_FILTERS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE;
    CUDA_CHECK(cudaMalloc(&d_temp_grad, conv1_size * sizeof(float)));
    relu_backward(cnn->conv1->d_output, cnn->pool1->d_input_gradients, d_temp_grad, conv1_size);

    blocks = (conv1_size + threads - 1) / threads;
    accumulate_gradients_kernel<<<blocks, threads>>>(cnn->d_conv1_relu_grad, d_temp_grad, conv1_size);
    CUDA_CHECK(cudaFree(d_temp_grad));

    // Backward through Conv1 (accumulate gradients)
    conv_backward(cnn->conv1, d_input, cnn->d_conv1_relu_grad, batch_size);

    CUDA_CHECK(cudaGetLastError());
}
