#include "cnn.cuh"
#include "forward.cuh"
#include "backward.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

// ==================== DECODER FORWARD PASS ====================

// Upsample forward kernel (nearest neighbor)
__global__ void upsample_forward_kernel(float* input, float* output,
                                       int batch_size, int channels,
                                       int input_size, int output_size, int scale) {
    int b = blockIdx.z;
    int c = blockIdx.y;
    int oh = blockIdx.x * blockDim.x + threadIdx.x;
    int ow = threadIdx.y;
    
    if (b < batch_size && c < channels && oh < output_size && ow < output_size) {
        int ih = oh / scale;
        int iw = ow / scale;
        
        int input_idx = b * (channels * input_size * input_size) +
                       c * (input_size * input_size) +
                       ih * input_size + iw;
        int output_idx = b * (channels * output_size * output_size) +
                        c * (output_size * output_size) +
                        oh * output_size + ow;
        
        output[output_idx] = input[input_idx];
    }
}

// Transpose convolution forward kernel
__global__ void transpose_conv_forward_kernel(float* input, float* weights, float* bias, float* output,
                                             int batch_size, int input_channels, int output_channels,
                                             int input_size, int output_size, int kernel_size,
                                             int stride, int padding) {
    int b = blockIdx.z;
    int oc = blockIdx.y;
    int oh = blockIdx.x * blockDim.x + threadIdx.x;
    int ow = threadIdx.y;
    
    if (b >= batch_size || oc >= output_channels || oh >= output_size || ow >= output_size)
        return;
    
    float sum = 0.0f;
    
    // For transpose conv, we need to find which input positions contribute to this output
    for (int ic = 0; ic < input_channels; ic++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // Calculate input position that maps to this output
                int ih = (oh + padding - kh) / stride;
                int iw = (ow + padding - kw) / stride;
                
                // Check if this input position is valid and maps exactly to output position
                if (ih >= 0 && ih < input_size && iw >= 0 && iw < input_size) {
                    if ((oh + padding - kh) % stride == 0 && (ow + padding - kw) % stride == 0) {
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
    }
    
    int output_idx = b * (output_channels * output_size * output_size) +
                    oc * (output_size * output_size) +
                    oh * output_size + ow;
    output[output_idx] = sum + bias[oc];
}

// Host wrapper for upsample forward
void upsample_forward(UpsampleLayer* layer, float* d_input, int batch_size) {
    dim3 blockDim(16, 16);
    dim3 gridDim((layer->output_size + 15) / 16, layer->channels, batch_size);
    
    upsample_forward_kernel<<<gridDim, blockDim>>>(
        d_input, layer->d_output,
        batch_size, layer->channels,
        layer->input_size, layer->output_size, layer->scale_factor
    );
    CUDA_CHECK(cudaGetLastError());
}

// Host wrapper for transpose conv forward
void transpose_conv_forward(TransposeConvLayer* layer, float* d_input, int batch_size) {
    dim3 blockDim(16, 16);
    dim3 gridDim((layer->output_size + 15) / 16, layer->output_channels, batch_size);
    
    transpose_conv_forward_kernel<<<gridDim, blockDim>>>(
        d_input, layer->d_weights, layer->d_bias, layer->d_output,
        batch_size, layer->input_channels, layer->output_channels,
        layer->input_size, layer->output_size, layer->kernel_size,
        layer->stride, layer->padding
    );
    CUDA_CHECK(cudaGetLastError());
}

// Decoder forward pass
void decoder_forward(Decoder* decoder, float* d_input, int batch_size) {
    // Note: batch_size parameter overrides decoder->batch_size if needed
    
    // Upsample1: 8×8 → 16×16
    upsample_forward(decoder->upsample1, d_input, batch_size);
    
    // TransConv1 + ReLU: 128ch → 64ch
    transpose_conv_forward(decoder->tconv1, decoder->upsample1->d_output, batch_size);
    relu_forward(decoder->tconv1->d_output, decoder->d_tconv1_relu,
                 batch_size * 64 * 16 * 16);
    
    // Upsample2: 16×16 → 32×32
    upsample_forward(decoder->upsample2, decoder->d_tconv1_relu, batch_size);
    
    // TransConv2: 64ch → 3ch (final reconstruction)
    transpose_conv_forward(decoder->tconv2, decoder->upsample2->d_output, batch_size);
    
    // Copy final output to reconstructed
    CUDA_CHECK(cudaMemcpy(decoder->d_reconstructed, decoder->tconv2->d_output,
                          batch_size * 3 * 32 * 32 * sizeof(float),
                          cudaMemcpyDeviceToDevice));
}

// ==================== DECODER BACKWARD PASS ====================

// Transpose convolution backward kernel
__global__ void transpose_conv_backward_kernel(float* input, float* weights, float* output_gradient,
                                              float* weight_gradients, float* bias_gradients, float* input_gradients,
                                              int batch_size, int input_channels, int output_channels,
                                              int input_size, int output_size, int kernel_size,
                                              int stride, int padding) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * input_channels * input_size * input_size;
    
    if (tid >= total_threads) return;
    
    int b = tid / (input_channels * input_size * input_size);
    int remainder = tid % (input_channels * input_size * input_size);
    int ic = remainder / (input_size * input_size);
    remainder = remainder % (input_size * input_size);
    int ih = remainder / input_size;
    int iw = remainder % input_size;
    
    float input_grad = 0.0f;
    float input_val = input[tid];
    
    for (int oc = 0; oc < output_channels; oc++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int oh = ih * stride + kh - padding;
                int ow = iw * stride + kw - padding;
                
                if (oh >= 0 && oh < output_size && ow >= 0 && ow < output_size) {
                    int output_idx = b * (output_channels * output_size * output_size) +
                                    oc * (output_size * output_size) +
                                    oh * output_size + ow;
                    int weight_idx = oc * (input_channels * kernel_size * kernel_size) +
                                    ic * (kernel_size * kernel_size) +
                                    kh * kernel_size + kw;
                    
                    float out_grad = output_gradient[output_idx];
                    
                    // Accumulate input gradient
                    input_grad += out_grad * weights[weight_idx];
                    
                    // Accumulate weight gradient
                    atomicAdd(&weight_gradients[weight_idx], out_grad * input_val);
                }
            }
        }
        
        // Bias gradient (only once per output channel per batch element)
        if (ic == 0 && ih == 0 && iw == 0) {
            for (int oh = 0; oh < output_size; oh++) {
                for (int ow = 0; ow < output_size; ow++) {
                    int output_idx = b * (output_channels * output_size * output_size) +
                                    oc * (output_size * output_size) +
                                    oh * output_size + ow;
                    atomicAdd(&bias_gradients[oc], output_gradient[output_idx]);
                }
            }
        }
    }
    
    input_gradients[tid] = input_grad;
}

// Upsample backward kernel
__global__ void upsample_backward_kernel(float* output_gradient, float* input_gradients,
                                        int batch_size, int channels,
                                        int input_size, int output_size, int scale) {
    int b = blockIdx.z;
    int c = blockIdx.y;
    int ih = blockIdx.x * blockDim.x + threadIdx.x;
    int iw = threadIdx.y;
    
    if (b >= batch_size || c >= channels || ih >= input_size || iw >= input_size)
        return;
    
    float grad_sum = 0.0f;
    
    // Sum gradients from all output positions that came from this input position
    for (int s_h = 0; s_h < scale; s_h++) {
        for (int s_w = 0; s_w < scale; s_w++) {
            int oh = ih * scale + s_h;
            int ow = iw * scale + s_w;
            
            if (oh < output_size && ow < output_size) {
                int output_idx = b * (channels * output_size * output_size) +
                                c * (output_size * output_size) +
                                oh * output_size + ow;
                grad_sum += output_gradient[output_idx];
            }
        }
    }
    
    int input_idx = b * (channels * input_size * input_size) +
                   c * (input_size * input_size) +
                   ih * input_size + iw;
    input_gradients[input_idx] = grad_sum;
}

// Host wrapper for transpose conv backward
void transpose_conv_backward(TransposeConvLayer* layer, float* d_input, float* d_output_gradient, int batch_size) {
    // Clear gradients
    CUDA_CHECK(cudaMemset(layer->d_weight_gradients, 0, 
                          layer->output_channels * layer->input_channels * 
                          layer->kernel_size * layer->kernel_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(layer->d_bias_gradients, 0, layer->output_channels * sizeof(float)));
    
    int total_threads = batch_size * layer->input_channels * layer->input_size * layer->input_size;
    int blockSize = 256;
    int gridSize = (total_threads + blockSize - 1) / blockSize;
    
    transpose_conv_backward_kernel<<<gridSize, blockSize>>>(
        d_input, layer->d_weights, d_output_gradient,
        layer->d_weight_gradients, layer->d_bias_gradients, layer->d_input_gradients,
        batch_size, layer->input_channels, layer->output_channels,
        layer->input_size, layer->output_size, layer->kernel_size,
        layer->stride, layer->padding
    );
    CUDA_CHECK(cudaGetLastError());
}

// Host wrapper for upsample backward
void upsample_backward(UpsampleLayer* layer, float* d_output_gradient, int batch_size) {
    dim3 blockDim(16, 16);
    dim3 gridDim((layer->input_size + 15) / 16, layer->channels, batch_size);
    
    upsample_backward_kernel<<<gridDim, blockDim>>>(
        d_output_gradient, layer->d_input_gradients,
        batch_size, layer->channels,
        layer->input_size, layer->output_size, layer->scale_factor
    );
    CUDA_CHECK(cudaGetLastError());
}

// Decoder backward pass
void decoder_backward(Decoder* decoder, float* d_output_gradient, float* d_input, int batch_size) {
    // d_output_gradient: gradient w.r.t. decoder output (reconstructed image)
    // d_input: original input to decoder (not used here, encoder output available in layers)
    
    // Backward through TransConv2
    transpose_conv_backward(decoder->tconv2, decoder->upsample2->d_output, d_output_gradient, batch_size);
    
    // Backward through Upsample2
    upsample_backward(decoder->upsample2, decoder->tconv2->d_input_gradients, batch_size);
    
    // Backward through ReLU
    relu_backward(decoder->tconv1->d_output, decoder->upsample2->d_input_gradients,
                  decoder->d_tconv1_relu_grad, batch_size * 64 * 16 * 16);
    
    // Backward through TransConv1
    transpose_conv_backward(decoder->tconv1, decoder->upsample1->d_output, decoder->d_tconv1_relu_grad, batch_size);
    
    // Backward through Upsample1
    upsample_backward(decoder->upsample1, decoder->tconv1->d_input_gradients, batch_size);
}

// Update decoder weights
void update_decoder_weights(Decoder* decoder, float learning_rate) {
    int blockSize = 256;
    
    // Update TransConv1 weights
    int tconv1_weight_size = 64 * 128 * 3 * 3;
    int gridSize = (tconv1_weight_size + blockSize - 1) / blockSize;
    update_weights_kernel<<<gridSize, blockSize>>>(decoder->tconv1->d_weights, 
                                                    decoder->tconv1->d_weight_gradients, 
                                                    learning_rate, tconv1_weight_size);
    
    gridSize = (64 + blockSize - 1) / blockSize;
    update_weights_kernel<<<gridSize, blockSize>>>(decoder->tconv1->d_bias, 
                                                    decoder->tconv1->d_bias_gradients, 
                                                    learning_rate, 64);
    
    // Update TransConv2 weights
    int tconv2_weight_size = 3 * 64 * 3 * 3;
    gridSize = (tconv2_weight_size + blockSize - 1) / blockSize;
    update_weights_kernel<<<gridSize, blockSize>>>(decoder->tconv2->d_weights, 
                                                    decoder->tconv2->d_weight_gradients, 
                                                    learning_rate, tconv2_weight_size);
    
    gridSize = (3 + blockSize - 1) / blockSize;
    update_weights_kernel<<<gridSize, blockSize>>>(decoder->tconv2->d_bias, 
                                                    decoder->tconv2->d_bias_gradients, 
                                                    learning_rate, 3);
    
    CUDA_CHECK(cudaGetLastError());
}
