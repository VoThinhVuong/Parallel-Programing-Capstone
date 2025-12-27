#include "backward.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


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
    gradient[idx] /= batch_size;  
}






__global__ void fc_bias_grad_kernel(const float* __restrict__ output_gradient,
                                   float* __restrict__ bias_gradients,
                                   int batch_size, int output_size) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= output_size) return;

    float sum = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        sum += output_gradient[b * output_size + o];
    }
    bias_gradients[o] = sum;
}



__global__ void fc_weight_grad_kernel(const float* __restrict__ input,
                                     const float* __restrict__ output_gradient,
                                     float* __restrict__ weight_gradients,
                                     int batch_size, int input_size, int output_size) {
    int o = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= output_size || i >= input_size) return;

    float sum = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        sum += output_gradient[b * output_size + o] * input[b * input_size + i];
    }
    weight_gradients[o * input_size + i] = sum;
}



__global__ void fc_input_grad_kernel(const float* __restrict__ weights,
                                    const float* __restrict__ output_gradient,
                                    float* __restrict__ input_gradients,
                                    int batch_size, int input_size, int output_size) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size || i >= input_size) return;

    float sum = 0.0f;
    
    for (int o = 0; o < output_size; o++) {
        sum += output_gradient[b * output_size + o] * weights[o * input_size + i];
    }
    input_gradients[b * input_size + i] = sum;
}


__global__ void relu_backward_kernel(float* input, float* output_gradient, float* input_gradient, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input_gradient[idx] = (input[idx] > 0) ? output_gradient[idx] : 0;
    }
}



__global__ void maxpool_backward_kernel(float* output_gradient, int* max_indices, float* input_gradients,
                                       int batch_size, int channels, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * output_size * output_size;
    
    if (idx >= total_outputs) {
        return;
    }
    
    
    int max_idx = max_indices[idx];
    input_gradients[max_idx] = output_gradient[idx];
}



__global__ void conv_backward_weights_kernel(float* input, float* output_gradient,
                                             float* weight_gradients, float* bias_gradients,
                                             int batch_size, int input_channels, int output_channels,
                                             int input_size, int output_size, int kernel_size,
                                             int stride, int padding) {
    int oc = blockIdx.z;
    int ic = blockIdx.y;
    int kh = blockIdx.x * blockDim.x + threadIdx.x;
    int kw = threadIdx.y;
    
    if (oc >= output_channels || ic >= input_channels || kh >= kernel_size || kw >= kernel_size) {
        return;
    }
    
    int weight_idx = oc * (input_channels * kernel_size * kernel_size) +
                     ic * (kernel_size * kernel_size) +
                     kh * kernel_size + kw;
    
    float weight_grad = 0.0f;
    float bias_grad = 0.0f;
    
    
    for (int b = 0; b < batch_size; b++) {
        for (int oh = 0; oh < output_size; oh++) {
            for (int ow = 0; ow < output_size; ow++) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;
                
                if (ih >= 0 && ih < input_size && iw >= 0 && iw < input_size) {
                    int output_idx = b * (output_channels * output_size * output_size) +
                                    oc * (output_size * output_size) +
                                    oh * output_size + ow;
                    int input_idx = b * (input_channels * input_size * input_size) +
                                   ic * (input_size * input_size) +
                                   ih * input_size + iw;
                    
                    weight_grad += output_gradient[output_idx] * input[input_idx];
                    
                    
                    if (ic == 0 && kh == 0 && kw == 0) {
                        bias_grad += output_gradient[output_idx];
                    }
                }
            }
        }
    }
    
    weight_gradients[weight_idx] = weight_grad;
    
    
    if (ic == 0 && kh == 0 && kw == 0) {
        bias_gradients[oc] = bias_grad;
    }
}


__global__ void conv_backward_input_kernel(float* output_gradient, float* weights,
                                          float* input_gradients,
                                          int batch_size, int input_channels, int output_channels,
                                          int input_size, int output_size, int kernel_size,
                                          int stride, int padding) {
    int b = blockIdx.z;
    int ic = blockIdx.y;
    int grid_w = (input_size + blockDim.y - 1) / blockDim.y;
    int ih_block = blockIdx.x / grid_w;
    int iw_block = blockIdx.x % grid_w;
    int ih = ih_block * blockDim.x + threadIdx.x;
    int iw = iw_block * blockDim.y + threadIdx.y;
    
    if (b >= batch_size || ic >= input_channels || ih >= input_size || iw >= input_size) {
        return;
    }
    
    int input_idx = b * (input_channels * input_size * input_size) +
                    ic * (input_size * input_size) +
                    ih * input_size + iw;
    
    float input_grad = 0.0f;
    
    
    for (int oc = 0; oc < output_channels; oc++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                
                int oh = (ih + padding - kh) / stride;
                int ow = (iw + padding - kw) / stride;
                
                
                if (oh >= 0 && oh < output_size && ow >= 0 && ow < output_size &&
                    (ih + padding - kh) % stride == 0 && (iw + padding - kw) % stride == 0) {
                    
                    int output_idx = b * (output_channels * output_size * output_size) +
                                    oc * (output_size * output_size) +
                                    oh * output_size + ow;
                    int weight_idx = oc * (input_channels * kernel_size * kernel_size) +
                                    ic * (kernel_size * kernel_size) +
                                    kh * kernel_size + kw;
                    
                    input_grad += output_gradient[output_idx] * weights[weight_idx];
                }
            }
        }
    }
    
    input_gradients[input_idx] = input_grad;
}


__global__ void update_weights_kernel(float* weights, float* gradients, float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}


__global__ void clear_gradients_kernel(float* gradients, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gradients[idx] = 0.0f;
    }
}

__global__ void accumulate_gradients_kernel(float* dst, const float* src, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] += src[idx];
    }
}


void softmax_cross_entropy_backward(float* d_output, uint8_t* d_labels, float* d_gradient,
                                    int batch_size, int num_classes) {
    dim3 block(16, 16);
    dim3 grid((batch_size + block.x - 1) / block.x, (num_classes + block.y - 1) / block.y);
    
    softmax_cross_entropy_backward_kernel<<<grid, block>>>(d_output, d_labels, d_gradient,
                                                           batch_size, num_classes);
    CUDA_CHECK(cudaGetLastError());
}

void fc_backward(FCLayer* layer, float* d_input, float* d_output_gradient, int batch_size) {
    
    int weight_size = layer->output_size * layer->input_size;
    int threads = 256;
    
    clear_gradients_kernel<<<(weight_size + threads - 1) / threads, threads>>>(layer->d_weight_gradients, weight_size);
    clear_gradients_kernel<<<(layer->output_size + threads - 1) / threads, threads>>>(layer->d_bias_gradients, layer->output_size);
    clear_gradients_kernel<<<(batch_size * layer->input_size + threads - 1) / threads, threads>>>(layer->d_input_gradients, batch_size * layer->input_size);
    
    
    
    int bias_blocks = (layer->output_size + threads - 1) / threads;
    fc_bias_grad_kernel<<<bias_blocks, threads>>>(
        d_output_gradient, layer->d_bias_gradients, batch_size, layer->output_size);
    CUDA_CHECK(cudaGetLastError());

    
    int wg_blocks_x = (layer->input_size + threads - 1) / threads;
    dim3 wg_grid(wg_blocks_x, layer->output_size);
    fc_weight_grad_kernel<<<wg_grid, threads>>>(
        d_input, d_output_gradient, layer->d_weight_gradients,
        batch_size, layer->input_size, layer->output_size);
    CUDA_CHECK(cudaGetLastError());

    
    int ig_blocks_x = (layer->input_size + threads - 1) / threads;
    dim3 ig_grid(ig_blocks_x, batch_size);
    fc_input_grad_kernel<<<ig_grid, threads>>>(
        layer->d_weights, d_output_gradient, layer->d_input_gradients,
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
    clear_gradients_kernel<<<(input_total_size + threads - 1) / threads, threads>>>(
        layer->d_input_gradients, input_total_size);
    
    
    int output_total_size = batch_size * layer->input_channels * layer->output_size * layer->output_size;
    maxpool_backward_kernel<<<(output_total_size + threads - 1) / threads, threads>>>(
        d_output_gradient, layer->d_max_indices, layer->d_input_gradients,
        batch_size, layer->input_channels, layer->output_size);
    CUDA_CHECK(cudaGetLastError());
}

void conv_backward(ConvLayer* layer, float* d_input, float* d_output_gradient, int batch_size) {
    
    dim3 weight_block(8, 8);
    dim3 weight_grid((layer->kernel_size + weight_block.x - 1) / weight_block.x,
                     layer->input_channels,
                     layer->output_channels);
    
    conv_backward_weights_kernel<<<weight_grid, weight_block>>>(
        d_input, d_output_gradient,
        layer->d_weight_gradients, layer->d_bias_gradients,
        batch_size, layer->input_channels, layer->output_channels,
        layer->input_size, layer->output_size, layer->kernel_size,
        layer->stride, layer->padding);
    CUDA_CHECK(cudaGetLastError());
    
    
    dim3 input_block(8, 8);
    int grid_h = (layer->input_size + input_block.x - 1) / input_block.x;
    int grid_w = (layer->input_size + input_block.y - 1) / input_block.y;
    dim3 input_grid(grid_h * grid_w, layer->input_channels, batch_size);
    
    conv_backward_input_kernel<<<input_grid, input_block>>>(
        d_output_gradient, layer->d_weights,
        layer->d_input_gradients,
        batch_size, layer->input_channels, layer->output_channels,
        layer->input_size, layer->output_size, layer->kernel_size,
        layer->stride, layer->padding);
    CUDA_CHECK(cudaGetLastError());
}


void backward_pass(CNN* cnn, float* d_input, uint8_t* d_labels) {
    int batch_size = cnn->batch_size;
    
    
    float* d_fc2_grad;
    CUDA_CHECK(cudaMalloc(&d_fc2_grad, batch_size * FC2_OUTPUT_SIZE * sizeof(float)));
    
    
    softmax_cross_entropy_backward(cnn->d_output, d_labels, d_fc2_grad, batch_size, FC2_OUTPUT_SIZE);
    
    
    fc_backward(cnn->fc2, cnn->d_fc1_relu, d_fc2_grad, batch_size);
    
    
    relu_backward(cnn->fc1->d_output, cnn->fc2->d_input_gradients, cnn->d_fc1_relu_grad,
                  batch_size * FC1_OUTPUT_SIZE);
    
    
    fc_backward(cnn->fc1, cnn->pool2->d_output, cnn->d_fc1_relu_grad, batch_size);
    
    
    maxpool_backward(cnn->pool2, cnn->fc1->d_input_gradients, batch_size);
    
    
    relu_backward(cnn->conv2->d_output, cnn->pool2->d_input_gradients, cnn->d_conv2_relu_grad,
                  batch_size * CONV2_FILTERS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE);
    
    
    conv_backward(cnn->conv2, cnn->pool1->d_output, cnn->d_conv2_relu_grad, batch_size);
    
    
    maxpool_backward(cnn->pool1, cnn->conv2->d_input_gradients, batch_size);
    
    
    relu_backward(cnn->conv1->d_output, cnn->pool1->d_input_gradients, cnn->d_conv1_relu_grad,
                  batch_size * CONV1_FILTERS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE);
    
    
    conv_backward(cnn->conv1, d_input, cnn->d_conv1_relu_grad, batch_size);
    
    CUDA_CHECK(cudaFree(d_fc2_grad));
}


void update_weights(CNN* cnn, float learning_rate) {
    update_encoder_weights(cnn, learning_rate);
    update_classifier_weights(cnn, learning_rate);
}

void update_classifier_weights(CNN* cnn, float learning_rate) {
    int threads = 256;

    
    int fc1_weight_size = FC1_OUTPUT_SIZE * FC1_INPUT_SIZE;
    update_weights_kernel<<<(fc1_weight_size + threads - 1) / threads, threads>>>(
        cnn->fc1->d_weights, cnn->fc1->d_weight_gradients, learning_rate, fc1_weight_size);
    update_weights_kernel<<<(FC1_OUTPUT_SIZE + threads - 1) / threads, threads>>>(
        cnn->fc1->d_bias, cnn->fc1->d_bias_gradients, learning_rate, FC1_OUTPUT_SIZE);

    
    int fc2_weight_size = FC2_OUTPUT_SIZE * FC2_INPUT_SIZE;
    update_weights_kernel<<<(fc2_weight_size + threads - 1) / threads, threads>>>(
        cnn->fc2->d_weights, cnn->fc2->d_weight_gradients, learning_rate, fc2_weight_size);
    update_weights_kernel<<<(FC2_OUTPUT_SIZE + threads - 1) / threads, threads>>>(
        cnn->fc2->d_bias, cnn->fc2->d_bias_gradients, learning_rate, FC2_OUTPUT_SIZE);

    CUDA_CHECK(cudaGetLastError());
}

void update_encoder_weights(CNN* cnn, float learning_rate) {
    int threads = 256;

    
    int conv1_weight_size = CONV1_FILTERS * INPUT_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE;
    update_weights_kernel<<<(conv1_weight_size + threads - 1) / threads, threads>>>(
        cnn->conv1->d_weights, cnn->conv1->d_weight_gradients, learning_rate, conv1_weight_size);
    update_weights_kernel<<<(CONV1_FILTERS + threads - 1) / threads, threads>>>(
        cnn->conv1->d_bias, cnn->conv1->d_bias_gradients, learning_rate, CONV1_FILTERS);

    
    int conv2_weight_size = CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE;
    update_weights_kernel<<<(conv2_weight_size + threads - 1) / threads, threads>>>(
        cnn->conv2->d_weights, cnn->conv2->d_weight_gradients, learning_rate, conv2_weight_size);
    update_weights_kernel<<<(CONV2_FILTERS + threads - 1) / threads, threads>>>(
        cnn->conv2->d_bias, cnn->conv2->d_bias_gradients, learning_rate, CONV2_FILTERS);

    CUDA_CHECK(cudaGetLastError());
}

void backprop_reconstruction_to_encoder(CNN* cnn, float* d_input, float* d_pool2_gradient, int batch_size) {
    
    

    static float* d_conv2_w_tmp = NULL;
    static float* d_conv2_b_tmp = NULL;
    static float* d_conv1_w_tmp = NULL;
    static float* d_conv1_b_tmp = NULL;

    const int conv2_weight_size = CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE;
    const int conv1_weight_size = CONV1_FILTERS * INPUT_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE;

    if (!d_conv2_w_tmp) {
        CUDA_CHECK(cudaMalloc(&d_conv2_w_tmp, conv2_weight_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv2_b_tmp, CONV2_FILTERS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv1_w_tmp, conv1_weight_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv1_b_tmp, CONV1_FILTERS * sizeof(float)));
    }

    
    maxpool_backward(cnn->pool2, d_pool2_gradient, batch_size);

    
    relu_backward(cnn->conv2->d_output, cnn->pool2->d_input_gradients, cnn->d_conv2_relu_grad,
                  batch_size * CONV2_FILTERS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE);

    
    float* conv2_w_orig = cnn->conv2->d_weight_gradients;
    float* conv2_b_orig = cnn->conv2->d_bias_gradients;
    cnn->conv2->d_weight_gradients = d_conv2_w_tmp;
    cnn->conv2->d_bias_gradients = d_conv2_b_tmp;
    conv_backward(cnn->conv2, cnn->pool1->d_output, cnn->d_conv2_relu_grad, batch_size);

    
    int threads = 256;
    accumulate_gradients_kernel<<<(conv2_weight_size + threads - 1) / threads, threads>>>(
        conv2_w_orig, d_conv2_w_tmp, conv2_weight_size);
    accumulate_gradients_kernel<<<(CONV2_FILTERS + threads - 1) / threads, threads>>>(
        conv2_b_orig, d_conv2_b_tmp, CONV2_FILTERS);
    CUDA_CHECK(cudaGetLastError());

    
    cnn->conv2->d_weight_gradients = conv2_w_orig;
    cnn->conv2->d_bias_gradients = conv2_b_orig;

    
    maxpool_backward(cnn->pool1, cnn->conv2->d_input_gradients, batch_size);

    
    relu_backward(cnn->conv1->d_output, cnn->pool1->d_input_gradients, cnn->d_conv1_relu_grad,
                  batch_size * CONV1_FILTERS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE);

    
    float* conv1_w_orig = cnn->conv1->d_weight_gradients;
    float* conv1_b_orig = cnn->conv1->d_bias_gradients;
    cnn->conv1->d_weight_gradients = d_conv1_w_tmp;
    cnn->conv1->d_bias_gradients = d_conv1_b_tmp;
    conv_backward(cnn->conv1, d_input, cnn->d_conv1_relu_grad, batch_size);

    accumulate_gradients_kernel<<<(conv1_weight_size + threads - 1) / threads, threads>>>(
        conv1_w_orig, d_conv1_w_tmp, conv1_weight_size);
    accumulate_gradients_kernel<<<(CONV1_FILTERS + threads - 1) / threads, threads>>>(
        conv1_b_orig, d_conv1_b_tmp, CONV1_FILTERS);
    CUDA_CHECK(cudaGetLastError());

    cnn->conv1->d_weight_gradients = conv1_w_orig;
    cnn->conv1->d_bias_gradients = conv1_b_orig;
}
