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

    int num_w_blocks = (output_size + blockDim.y - 1) / blockDim.y;
    int oh_block = blockIdx.x / num_w_blocks;
    int ow_block = blockIdx.x % num_w_blocks;

    int oh = oh_block * blockDim.x + threadIdx.x;
    int ow = ow_block * blockDim.y + threadIdx.y;

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

// Host wrapper for upsample forward
void upsample_forward(UpsampleLayer* layer, float* d_input, int batch_size) {
    dim3 blockDim(8, 8);
    int grid_h = (layer->output_size + blockDim.x - 1) / blockDim.x;
    int grid_w = (layer->output_size + blockDim.y - 1) / blockDim.y;
    dim3 gridDim(grid_h * grid_w, layer->channels, batch_size);

    upsample_forward_kernel<<<gridDim, blockDim>>>(
        d_input, layer->d_output,
        batch_size, layer->channels,
        layer->input_size, layer->output_size, layer->scale_factor);
    CUDA_CHECK(cudaGetLastError());
}

// Decoder forward pass
void decoder_forward(Decoder* decoder, float* d_input, int batch_size) {
    // Note: batch_size parameter overrides decoder->batch_size if needed
    
    // Conv1 + ReLU: 128ch → 128ch, 8×8
    conv_forward(decoder->conv1, d_input, batch_size);
    relu_forward(decoder->conv1->d_output, decoder->d_conv1_relu,
                 batch_size * 128 * 8 * 8);
    
    // Upsample1: 8×8 → 16×16 (128 channels)
    upsample_forward(decoder->upsample1, decoder->d_conv1_relu, batch_size);
    
    // Conv2 + ReLU: 128ch → 256ch, 16×16
    conv_forward(decoder->conv2, decoder->upsample1->d_output, batch_size);
    relu_forward(decoder->conv2->d_output, decoder->d_conv2_relu,
                 batch_size * 256 * 16 * 16);
    
    // Upsample2: 16×16 → 32×32 (256 channels)
    upsample_forward(decoder->upsample2, decoder->d_conv2_relu, batch_size);
    
    // Conv3: 256ch → 3ch, 32×32 (final reconstruction)
    conv_forward(decoder->conv3, decoder->upsample2->d_output, batch_size);
    
    // Copy final output to reconstructed
    CUDA_CHECK(cudaMemcpy(decoder->d_reconstructed, decoder->conv3->d_output,
                          batch_size * 3 * 32 * 32 * sizeof(float),
                          cudaMemcpyDeviceToDevice));
}

// ==================== DECODER BACKWARD PASS ====================

// Upsample backward kernel
__global__ void upsample_backward_kernel(float* output_gradient, float* input_gradients,
                                        int batch_size, int channels,
                                        int input_size, int output_size, int scale) {
    int b = blockIdx.z;
    int c = blockIdx.y;

    int num_w_blocks = (input_size + blockDim.y - 1) / blockDim.y;
    int ih_block = blockIdx.x / num_w_blocks;
    int iw_block = blockIdx.x % num_w_blocks;

    int ih = ih_block * blockDim.x + threadIdx.x;
    int iw = iw_block * blockDim.y + threadIdx.y;

    if (b >= batch_size || c >= channels || ih >= input_size || iw >= input_size) {
        return;
    }

    float grad_sum = 0.0f;

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

// Host wrapper for upsample backward
void upsample_backward(UpsampleLayer* layer, float* d_output_gradient, int batch_size) {
    dim3 blockDim(8, 8);
    int grid_h = (layer->input_size + blockDim.x - 1) / blockDim.x;
    int grid_w = (layer->input_size + blockDim.y - 1) / blockDim.y;
    dim3 gridDim(grid_h * grid_w, layer->channels, batch_size);

    upsample_backward_kernel<<<gridDim, blockDim>>>(
        d_output_gradient, layer->d_input_gradients,
        batch_size, layer->channels,
        layer->input_size, layer->output_size, layer->scale_factor);
    CUDA_CHECK(cudaGetLastError());
}

// Decoder backward pass
void decoder_backward(Decoder* decoder, float* d_output_gradient, float* d_input, int batch_size) {
    // d_output_gradient: gradient w.r.t. decoder output (reconstructed image)
    // d_input: original input to decoder (not used here, encoder output available in layers)
    
    // Backward through Conv3
    conv_backward(decoder->conv3, decoder->upsample2->d_output, d_output_gradient, batch_size);
    
    // Backward through Upsample2
    upsample_backward(decoder->upsample2, decoder->conv3->d_input_gradients, batch_size);
    
    // Backward through ReLU for Conv2
    relu_backward(decoder->conv2->d_output, decoder->upsample2->d_input_gradients,
                  decoder->d_conv2_relu_grad, batch_size * 256 * 16 * 16);
    
    // Backward through Conv2
    conv_backward(decoder->conv2, decoder->upsample1->d_output, decoder->d_conv2_relu_grad, batch_size);
    
    // Backward through Upsample1
    upsample_backward(decoder->upsample1, decoder->conv2->d_input_gradients, batch_size);
    
    // Backward through ReLU for Conv1
    relu_backward(decoder->conv1->d_output, decoder->upsample1->d_input_gradients,
                  decoder->d_conv1_relu_grad, batch_size * 128 * 8 * 8);
    
    // Backward through Conv1
    conv_backward(decoder->conv1, d_input, decoder->d_conv1_relu_grad, batch_size);
}

// Update decoder weights
void update_decoder_weights(Decoder* decoder, float learning_rate) {
    int blockSize = 256;
    int gridSize;
    
    // Update Conv1 weights: 128 * 128 * 3 * 3
    int conv1_weight_size = 128 * 128 * 3 * 3;
    gridSize = (conv1_weight_size + blockSize - 1) / blockSize;
    update_weights_kernel<<<gridSize, blockSize>>>(decoder->conv1->d_weights, 
                                                    decoder->conv1->d_weight_gradients, 
                                                    learning_rate, conv1_weight_size);
    
    gridSize = (128 + blockSize - 1) / blockSize;
    update_weights_kernel<<<gridSize, blockSize>>>(decoder->conv1->d_bias, 
                                                    decoder->conv1->d_bias_gradients, 
                                                    learning_rate, 128);
    
    // Update Conv2 weights: 256 * 128 * 3 * 3
    int conv2_weight_size = 256 * 128 * 3 * 3;
    gridSize = (conv2_weight_size + blockSize - 1) / blockSize;
    update_weights_kernel<<<gridSize, blockSize>>>(decoder->conv2->d_weights, 
                                                    decoder->conv2->d_weight_gradients, 
                                                    learning_rate, conv2_weight_size);
    
    gridSize = (256 + blockSize - 1) / blockSize;
    update_weights_kernel<<<gridSize, blockSize>>>(decoder->conv2->d_bias, 
                                                    decoder->conv2->d_bias_gradients, 
                                                    learning_rate, 256);
    
    // Update Conv3 weights: 3 * 256 * 3 * 3
    int conv3_weight_size = 3 * 256 * 3 * 3;
    gridSize = (conv3_weight_size + blockSize - 1) / blockSize;
    update_weights_kernel<<<gridSize, blockSize>>>(decoder->conv3->d_weights, 
                                                    decoder->conv3->d_weight_gradients, 
                                                    learning_rate, conv3_weight_size);
    
    gridSize = (3 + blockSize - 1) / blockSize;
    update_weights_kernel<<<gridSize, blockSize>>>(decoder->conv3->d_bias, 
                                                    decoder->conv3->d_bias_gradients, 
                                                    learning_rate, 3);
    
    CUDA_CHECK(cudaGetLastError());
}
