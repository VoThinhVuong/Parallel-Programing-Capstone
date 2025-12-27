#include "cnn.cuh"
#include "forward.cuh"
#include "backward.cuh"
#include <cuda_runtime.h>
#include <stdio.h>




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


__global__ void transpose_conv_forward_kernel(float* input, float* weights, float* bias, float* output,
                                             int batch_size, int input_channels, int output_channels,
                                             int input_size, int output_size, int kernel_size,
                                             int stride, int padding) {
    int b = blockIdx.z;
    int oc = blockIdx.y;

    int num_w_blocks = (output_size + blockDim.y - 1) / blockDim.y;
    int oh_block = blockIdx.x / num_w_blocks;
    int ow_block = blockIdx.x % num_w_blocks;

    int oh = oh_block * blockDim.x + threadIdx.x;
    int ow = ow_block * blockDim.y + threadIdx.y;

    if (b >= batch_size || oc >= output_channels || oh >= output_size || ow >= output_size) {
        return;
    }

    float sum = bias[oc];

    extern __shared__ float shared_input[];

    int tile_height = blockDim.x + kernel_size - 1;
    int tile_width = blockDim.y + kernel_size - 1;

    for (int ic = 0; ic < input_channels; ic++) {
        
        int load_h = (tile_height + blockDim.x - 1) / blockDim.x;
        int load_w = (tile_width + blockDim.y - 1) / blockDim.y;

        for (int lh = 0; lh < load_h; lh++) {
            for (int lw = 0; lw < load_w; lw++) {
                int sh = threadIdx.x + lh * blockDim.x;
                int sw = threadIdx.y + lw * blockDim.y;

                if (sh < tile_height && sw < tile_width) {
                    int ih = oh_block * blockDim.x + sh - padding;
                    int iw = ow_block * blockDim.y + sw - padding;

                    if (ih >= 0 && ih < input_size && iw >= 0 && iw < input_size) {
                        int input_idx = b * (input_channels * input_size * input_size) +
                                        ic * (input_size * input_size) +
                                        ih * input_size + iw;
                        shared_input[sh * tile_width + sw] = input[input_idx];
                    } else {
                        shared_input[sh * tile_width + sw] = 0.0f;
                    }
                }
            }
        }

        __syncthreads();

        
        
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int sh = threadIdx.x + 2 * padding - kh;
                int sw = threadIdx.y + 2 * padding - kw;

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


void transpose_conv_forward(TransposeConvLayer* layer, float* d_input, int batch_size) {
    dim3 blockDim(8, 8);
    int grid_h = (layer->output_size + blockDim.x - 1) / blockDim.x;
    int grid_w = (layer->output_size + blockDim.y - 1) / blockDim.y;
    dim3 gridDim(grid_h * grid_w, layer->output_channels, batch_size);

    int tile_height = blockDim.x + layer->kernel_size - 1;
    int tile_width = blockDim.y + layer->kernel_size - 1;
    int shared_mem_size = tile_height * tile_width * sizeof(float);

    transpose_conv_forward_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        d_input, layer->d_weights, layer->d_bias, layer->d_output,
        batch_size, layer->input_channels, layer->output_channels,
        layer->input_size, layer->output_size, layer->kernel_size,
        layer->stride, layer->padding);
    CUDA_CHECK(cudaGetLastError());
}


void decoder_forward(Decoder* decoder, float* d_input, int batch_size) {
    
    upsample_forward(decoder->upsample1, d_input, batch_size);

    
    transpose_conv_forward(decoder->tconv1, decoder->upsample1->d_output, batch_size);
    relu_forward(decoder->tconv1->d_output, decoder->d_tconv1_relu,
                 batch_size * 64 * 16 * 16);

    
    upsample_forward(decoder->upsample2, decoder->d_tconv1_relu, batch_size);

    
    transpose_conv_forward(decoder->tconv2, decoder->upsample2->d_output, batch_size);

    CUDA_CHECK(cudaMemcpy(decoder->d_reconstructed, decoder->tconv2->d_output,
                          batch_size * 3 * 32 * 32 * sizeof(float),
                          cudaMemcpyDeviceToDevice));
}



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

                    input_grad += out_grad * weights[weight_idx];
                    atomicAdd(&weight_gradients[weight_idx], out_grad * input_val);
                }
            }
        }

        
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

void transpose_conv_backward(TransposeConvLayer* layer, float* d_input, float* d_output_gradient, int batch_size) {
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
        layer->stride, layer->padding);
    CUDA_CHECK(cudaGetLastError());
}

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

void decoder_backward(Decoder* decoder, float* d_output_gradient, float* d_input, int batch_size) {
    (void)d_input;

    
    transpose_conv_backward(decoder->tconv2, decoder->upsample2->d_output, d_output_gradient, batch_size);

    
    upsample_backward(decoder->upsample2, decoder->tconv2->d_input_gradients, batch_size);

    
    relu_backward(decoder->tconv1->d_output, decoder->upsample2->d_input_gradients,
                  decoder->d_tconv1_relu_grad, batch_size * 64 * 16 * 16);

    
    transpose_conv_backward(decoder->tconv1, decoder->upsample1->d_output, decoder->d_tconv1_relu_grad, batch_size);

    
    upsample_backward(decoder->upsample1, decoder->tconv1->d_input_gradients, batch_size);
}

void update_decoder_weights(Decoder* decoder, float learning_rate) {
    int threads = 256;

    
    int tconv1_weight_size = decoder->tconv1->output_channels * decoder->tconv1->input_channels *
                             decoder->tconv1->kernel_size * decoder->tconv1->kernel_size;
    update_weights_kernel<<<(tconv1_weight_size + threads - 1) / threads, threads>>>(
        decoder->tconv1->d_weights, decoder->tconv1->d_weight_gradients, learning_rate, tconv1_weight_size);
    update_weights_kernel<<<(decoder->tconv1->output_channels + threads - 1) / threads, threads>>>(
        decoder->tconv1->d_bias, decoder->tconv1->d_bias_gradients, learning_rate, decoder->tconv1->output_channels);

    
    int tconv2_weight_size = decoder->tconv2->output_channels * decoder->tconv2->input_channels *
                             decoder->tconv2->kernel_size * decoder->tconv2->kernel_size;
    update_weights_kernel<<<(tconv2_weight_size + threads - 1) / threads, threads>>>(
        decoder->tconv2->d_weights, decoder->tconv2->d_weight_gradients, learning_rate, tconv2_weight_size);
    update_weights_kernel<<<(decoder->tconv2->output_channels + threads - 1) / threads, threads>>>(
        decoder->tconv2->d_bias, decoder->tconv2->d_bias_gradients, learning_rate, decoder->tconv2->output_channels);

    CUDA_CHECK(cudaGetLastError());
}
