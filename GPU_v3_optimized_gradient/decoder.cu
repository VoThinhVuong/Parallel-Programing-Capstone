#include "cnn.cuh"
#include "forward.cuh"
#include "backward.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>






__global__ void upsample_forward_kernel(float* input, float* output,
                                       int batch_size, int channels,
                                       int input_size, int output_size, int scale) {
    int c = blockIdx.x;
    int b = blockIdx.y;
    int oh = threadIdx.x;
    int ow = threadIdx.y;

    if (b >= batch_size || c >= channels || oh >= output_size || ow >= output_size) return;

    int ih = oh / scale;
    int iw = ow / scale;

    float v = input[((b * channels + c) * input_size + ih) * input_size + iw];
    output[((b * channels + c) * output_size + oh) * output_size + ow] = v;
}

void upsample_forward(UpsampleLayer* layer, float* d_input, int batch_size) {
    dim3 block(layer->output_size, layer->output_size);
    dim3 grid(layer->channels, batch_size);
    upsample_forward_kernel<<<grid, block>>>(
        d_input, layer->d_output,
        batch_size, layer->channels,
        layer->input_size, layer->output_size, layer->scale_factor);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void upsample_backward_kernel(float* output_gradient, float* input_gradients,
                                        int batch_size, int channels,
                                        int input_size, int output_size, int scale) {
    int c = blockIdx.x;
    int b = blockIdx.y;
    int ih = threadIdx.x;
    int iw = threadIdx.y;

    if (b >= batch_size || c >= channels || ih >= input_size || iw >= input_size) return;

    float grad_sum = 0.0f;
    int oh0 = ih * scale;
    int ow0 = iw * scale;
    for (int dh = 0; dh < scale; dh++) {
        for (int dw = 0; dw < scale; dw++) {
            int oh = oh0 + dh;
            int ow = ow0 + dw;
            if (oh < output_size && ow < output_size) {
                grad_sum += output_gradient[((b * channels + c) * output_size + oh) * output_size + ow];
            }
        }
    }

    input_gradients[((b * channels + c) * input_size + ih) * input_size + iw] = grad_sum;
}

void upsample_backward(UpsampleLayer* layer, float* d_output_gradient, int batch_size) {
    dim3 block(layer->input_size, layer->input_size);
    dim3 grid(layer->channels, batch_size);
    upsample_backward_kernel<<<grid, block>>>(
        d_output_gradient, layer->d_input_gradients,
        batch_size, layer->channels,
        layer->input_size, layer->output_size, layer->scale_factor);
    CUDA_CHECK(cudaGetLastError());
}



#ifndef TC_TILE
#define TC_TILE 8
#endif

__global__ void transpose_conv_forward_kernel(float* input, float* weights, float* bias, float* output,
                                             int batch_size, int input_channels, int output_channels,
                                             int input_size, int output_size, int kernel_size,
                                             int stride, int padding) {
    int oc = blockIdx.x;
    int b = blockIdx.y;

    int grid_w = (output_size + TC_TILE - 1) / TC_TILE;
    int tile_id = blockIdx.z;
    int tile_h = tile_id / grid_w;
    int tile_w = tile_id % grid_w;
    int oh0 = tile_h * TC_TILE;
    int ow0 = tile_w * TC_TILE;

    int th = threadIdx.x;
    int tw = threadIdx.y;
    int oh = oh0 + th;
    int ow = ow0 + tw;

    if (b >= batch_size || oc >= output_channels || th >= TC_TILE || tw >= TC_TILE) return;

    float sum = 0.0f;
    if (oh < output_size && ow < output_size) {
        sum = bias[oc];
    }

    
    extern __shared__ float shared_input[];
    int tile_h_in = TC_TILE + kernel_size - 1;
    int tile_w_in = TC_TILE + kernel_size - 1;

    for (int ic = 0; ic < input_channels; ic++) {
        
        for (int sh = th; sh < tile_h_in; sh += TC_TILE) {
            for (int sw = tw; sw < tile_w_in; sw += TC_TILE) {
                int ih = oh0 - padding + sh;
                int iw = ow0 - padding + sw;
                float v = 0.0f;
                if (ih >= 0 && ih < input_size && iw >= 0 && iw < input_size) {
                    v = input[((b * input_channels + ic) * input_size + ih) * input_size + iw];
                }
                shared_input[sh * tile_w_in + sw] = v;
            }
        }
        __syncthreads();

        if (oh < output_size && ow < output_size) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    
                    
                    int sh = th + (2 * padding) - kh;
                    int sw = tw + (2 * padding) - kw;
                    if (sh >= 0 && sh < tile_h_in && sw >= 0 && sw < tile_w_in) {
                        int w_idx = oc * (input_channels * kernel_size * kernel_size) +
                                    ic * (kernel_size * kernel_size) +
                                    kh * kernel_size + kw;
                        sum += shared_input[sh * tile_w_in + sw] * weights[w_idx];
                    }
                }
            }
        }

        __syncthreads();
    }

    if (oh < output_size && ow < output_size) {
        output[((b * output_channels + oc) * output_size + oh) * output_size + ow] = sum;
    }
}

void transpose_conv_forward(TransposeConvLayer* layer, float* d_input, int batch_size) {
    int grid_h = (layer->output_size + TC_TILE - 1) / TC_TILE;
    int grid_w = (layer->output_size + TC_TILE - 1) / TC_TILE;

    dim3 block(TC_TILE, TC_TILE);
    dim3 grid(layer->output_channels, batch_size, grid_h * grid_w);

    int tile_h_in = TC_TILE + layer->kernel_size - 1;
    int tile_w_in = TC_TILE + layer->kernel_size - 1;
    size_t shared_mem_size = (size_t)tile_h_in * (size_t)tile_w_in * sizeof(float);

    transpose_conv_forward_kernel<<<grid, block, shared_mem_size>>>(
        d_input, layer->d_weights, layer->d_bias, layer->d_output,
        batch_size, layer->input_channels, layer->output_channels,
        layer->input_size, layer->output_size, layer->kernel_size,
        layer->stride, layer->padding);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void transpose_conv_backward_kernel(float* input, float* weights, float* output_gradient,
                                              float* weight_gradients, float* bias_gradients, float* input_gradients,
                                              int batch_size, int input_channels, int output_channels,
                                              int input_size, int output_size, int kernel_size,
                                              int stride, int padding) {
    
    
    (void)input; (void)weights; (void)output_gradient;
    (void)weight_gradients; (void)bias_gradients; (void)input_gradients;
    (void)batch_size; (void)input_channels; (void)output_channels;
    (void)input_size; (void)output_size; (void)kernel_size;
    (void)stride; (void)padding;
}



__global__ void transpose_conv_weight_bias_grad_kernel(const float* __restrict__ input,
                                                      const float* __restrict__ output_gradient,
                                                      float* __restrict__ weight_gradients,
                                                      float* __restrict__ bias_gradients,
                                                      int batch_size, int input_channels, int output_channels,
                                                      int input_size, int output_size,
                                                      int kernel_size, int padding) {
    int oc = blockIdx.z;
    int ic = blockIdx.y;
    int kh = blockIdx.x * blockDim.x + threadIdx.x;
    int kw = threadIdx.y;

    if (oc >= output_channels || ic >= input_channels || kh >= kernel_size || kw >= kernel_size) return;

    int w_idx = oc * (input_channels * kernel_size * kernel_size) +
                ic * (kernel_size * kernel_size) +
                kh * kernel_size + kw;

    float w_sum = 0.0f;
    float b_sum = 0.0f;

    for (int b = 0; b < batch_size; b++) {
        for (int oh = 0; oh < output_size; oh++) {
            for (int ow = 0; ow < output_size; ow++) {
                
                int ih = oh + padding - kh;
                int iw = ow + padding - kw;
                if (ih >= 0 && ih < input_size && iw >= 0 && iw < input_size) {
                    int out_idx = ((b * output_channels + oc) * output_size + oh) * output_size + ow;
                    int in_idx  = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    float out_g = output_gradient[out_idx];
                    w_sum += out_g * input[in_idx];

                    if (ic == 0 && kh == 0 && kw == 0) {
                        b_sum += out_g;
                    }
                }
            }
        }
    }

    weight_gradients[w_idx] = w_sum;
    if (ic == 0 && kh == 0 && kw == 0) {
        bias_gradients[oc] = b_sum;
    }
}



__global__ void transpose_conv_input_grad_kernel(const float* __restrict__ weights,
                                                const float* __restrict__ output_gradient,
                                                float* __restrict__ input_gradients,
                                                int batch_size, int input_channels, int output_channels,
                                                int input_size, int output_size,
                                                int kernel_size, int padding) {
    int b = blockIdx.z;
    int ic = blockIdx.y;

    int grid_w = (input_size + blockDim.y - 1) / blockDim.y;
    int ih_block = blockIdx.x / grid_w;
    int iw_block = blockIdx.x % grid_w;
    int ih = ih_block * blockDim.x + threadIdx.x;
    int iw = iw_block * blockDim.y + threadIdx.y;

    if (b >= batch_size || ic >= input_channels || ih >= input_size || iw >= input_size) return;

    float in_sum = 0.0f;
    for (int oc = 0; oc < output_channels; oc++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                
                int oh = ih - padding + kh;
                int ow = iw - padding + kw;
                if (oh >= 0 && oh < output_size && ow >= 0 && ow < output_size) {
                    int out_idx = ((b * output_channels + oc) * output_size + oh) * output_size + ow;
                    int w_idx = oc * (input_channels * kernel_size * kernel_size) +
                                ic * (kernel_size * kernel_size) +
                                kh * kernel_size + kw;
                    in_sum += output_gradient[out_idx] * weights[w_idx];
                }
            }
        }
    }

    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
    input_gradients[in_idx] = in_sum;
}

void transpose_conv_backward(TransposeConvLayer* layer, float* d_input, float* d_output_gradient, int batch_size) {
    
    dim3 wg_block(16, 16);
    dim3 wg_grid((layer->kernel_size + wg_block.x - 1) / wg_block.x,
                 layer->input_channels,
                 layer->output_channels);
    transpose_conv_weight_bias_grad_kernel<<<wg_grid, wg_block>>>(
        d_input, d_output_gradient,
        layer->d_weight_gradients, layer->d_bias_gradients,
        batch_size, layer->input_channels, layer->output_channels,
        layer->input_size, layer->output_size,
        layer->kernel_size, layer->padding);
    CUDA_CHECK(cudaGetLastError());

    
    dim3 ig_block(16, 16);
    int grid_h = (layer->input_size + ig_block.x - 1) / ig_block.x;
    int grid_w = (layer->input_size + ig_block.y - 1) / ig_block.y;
    dim3 ig_grid(grid_h * grid_w, layer->input_channels, batch_size);
    transpose_conv_input_grad_kernel<<<ig_grid, ig_block>>>(
        layer->d_weights, d_output_gradient,
        layer->d_input_gradients,
        batch_size, layer->input_channels, layer->output_channels,
        layer->input_size, layer->output_size,
        layer->kernel_size, layer->padding);
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

void decoder_backward(Decoder* decoder, float* d_output_gradient, float* d_input, int batch_size) {
    
    

    transpose_conv_backward(decoder->tconv2, decoder->upsample2->d_output, d_output_gradient, batch_size);
    upsample_backward(decoder->upsample2, decoder->tconv2->d_input_gradients, batch_size);

    relu_backward(decoder->tconv1->d_output, decoder->upsample2->d_input_gradients,
                  decoder->d_tconv1_relu_grad,
                  batch_size * 64 * 16 * 16);

    transpose_conv_backward(decoder->tconv1, decoder->upsample1->d_output, decoder->d_tconv1_relu_grad, batch_size);
    upsample_backward(decoder->upsample1, decoder->tconv1->d_input_gradients, batch_size);

    CUDA_CHECK(cudaMemcpy(d_input, decoder->upsample1->d_input_gradients,
                          batch_size * 128 * 8 * 8 * sizeof(float),
                          cudaMemcpyDeviceToDevice));
}

void update_decoder_weights(Decoder* decoder, float learning_rate) {
    int threads = 256;

    int tconv1_w = decoder->tconv1->output_channels * decoder->tconv1->input_channels *
                   decoder->tconv1->kernel_size * decoder->tconv1->kernel_size;
    update_weights_kernel<<<(tconv1_w + threads - 1) / threads, threads>>>(
        decoder->tconv1->d_weights, decoder->tconv1->d_weight_gradients, learning_rate, tconv1_w);
    update_weights_kernel<<<(decoder->tconv1->output_channels + threads - 1) / threads, threads>>>(
        decoder->tconv1->d_bias, decoder->tconv1->d_bias_gradients, learning_rate, decoder->tconv1->output_channels);

    int tconv2_w = decoder->tconv2->output_channels * decoder->tconv2->input_channels *
                   decoder->tconv2->kernel_size * decoder->tconv2->kernel_size;
    update_weights_kernel<<<(tconv2_w + threads - 1) / threads, threads>>>(
        decoder->tconv2->d_weights, decoder->tconv2->d_weight_gradients, learning_rate, tconv2_w);
    update_weights_kernel<<<(decoder->tconv2->output_channels + threads - 1) / threads, threads>>>(
        decoder->tconv2->d_bias, decoder->tconv2->d_bias_gradients, learning_rate, decoder->tconv2->output_channels);

    CUDA_CHECK(cudaGetLastError());
}
