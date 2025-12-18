#include "cnn.cuh"
#include "forward.cuh"
#include "backward.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Decoder implementation for GPU_v3_optimized_gradient.
// Matches Decoder/UpsampleLayer/TransposeConvLayer structs in cnn.cuh.

// ------------------------ Upsample ------------------------

// 1D kernels (flattened) tend to schedule better than (channels,batch) grids with
// large 2D blocks (e.g., 32x32 = 1024 threads) for these lightweight operations.

__global__ void upsample_forward_kernel_1d(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          int total_out,
                                          int channels,
                                          int input_size,
                                          int output_size,
                                          int scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

    int ow = idx % output_size;
    int tmp = idx / output_size;
    int oh = tmp % output_size;
    tmp /= output_size;
    int c = tmp % channels;
    int b = tmp / channels;

    int ih = oh / scale;
    int iw = ow / scale;

    int in_idx = ((b * channels + c) * input_size + ih) * input_size + iw;
    int out_idx = ((b * channels + c) * output_size + oh) * output_size + ow;
    output[out_idx] = input[in_idx];
}

void upsample_forward(UpsampleLayer* layer, float* d_input, int batch_size) {
    int threads = 256;
    int total_out = batch_size * layer->channels * layer->output_size * layer->output_size;
    int blocks = (total_out + threads - 1) / threads;
    upsample_forward_kernel_1d<<<blocks, threads>>>(
        d_input, layer->d_output,
        total_out,
        layer->channels,
        layer->input_size, layer->output_size, layer->scale_factor);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void upsample_backward_kernel_1d(const float* __restrict__ output_gradient,
                                           float* __restrict__ input_gradients,
                                           int total_in,
                                           int channels,
                                           int input_size,
                                           int output_size,
                                           int scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_in) return;

    int iw = idx % input_size;
    int tmp = idx / input_size;
    int ih = tmp % input_size;
    tmp /= input_size;
    int c = tmp % channels;
    int b = tmp / channels;

    float grad_sum = 0.0f;
    int oh0 = ih * scale;
    int ow0 = iw * scale;
    int base = (b * channels + c) * output_size * output_size;
    for (int dh = 0; dh < scale; dh++) {
        for (int dw = 0; dw < scale; dw++) {
            int oh = oh0 + dh;
            int ow = ow0 + dw;
            if (oh < output_size && ow < output_size) {
                grad_sum += output_gradient[base + oh * output_size + ow];
            }
        }
    }

    input_gradients[((b * channels + c) * input_size + ih) * input_size + iw] = grad_sum;
}

void upsample_backward(UpsampleLayer* layer, float* d_output_gradient, int batch_size) {
    int threads = 256;
    int total_in = batch_size * layer->channels * layer->input_size * layer->input_size;
    int blocks = (total_in + threads - 1) / threads;
    upsample_backward_kernel_1d<<<blocks, threads>>>(
        d_output_gradient, layer->d_input_gradients,
        total_in,
        layer->channels,
        layer->input_size, layer->output_size, layer->scale_factor);
    CUDA_CHECK(cudaGetLastError());
}

// ------------------------ Decoder -------------------------

void decoder_forward(Decoder* decoder, float* d_input, int batch_size) {
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
