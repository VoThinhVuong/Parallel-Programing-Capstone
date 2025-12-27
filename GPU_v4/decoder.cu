#include "cnn.cuh"
#include "forward.cuh"
#include "backward.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>









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




__global__ void transpose_conv_weight_bias_grad_ocic_reduce_kernel(const float* __restrict__ input,
                                                                  const float* __restrict__ output_gradient,
                                                                  float* __restrict__ weight_gradients,
                                                                  float* __restrict__ bias_gradients,
                                                                  int batch_size, int input_channels, int output_channels,
                                                                  int input_size, int output_size,
                                                                  int kernel_size, int padding) {
    int ocic = blockIdx.x;
    int oc = ocic / input_channels;
    int ic = ocic - oc * input_channels;
    if (oc >= output_channels || ic >= input_channels) return;

    int kk = kernel_size * kernel_size;
    float w_acc[9];
#pragma unroll
    for (int i = 0; i < 9; i++) w_acc[i] = 0.0f;
    float b_acc = 0.0f;

    int n = batch_size * output_size * output_size;
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        int ow = t % output_size;
        int tmp = t / output_size;
        int oh = tmp % output_size;
        int b = tmp / output_size;

        int out_idx = ((b * output_channels + oc) * output_size + oh) * output_size + ow;
        float out_g = output_gradient[out_idx];

        
        
        int ih0 = oh + padding;
        int iw0 = ow + padding;

        
        {
            int ih = ih0 - 0;
            if (ih >= 0 && ih < input_size) {
                int iw;
                iw = iw0 - 0;
                if (iw >= 0 && iw < input_size) {
                    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    w_acc[0] += out_g * input[in_idx];
                }
                iw = iw0 - 1;
                if (iw >= 0 && iw < input_size) {
                    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    w_acc[1] += out_g * input[in_idx];
                }
                iw = iw0 - 2;
                if (iw >= 0 && iw < input_size) {
                    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    w_acc[2] += out_g * input[in_idx];
                }
            }
        }
        
        {
            int ih = ih0 - 1;
            if (ih >= 0 && ih < input_size) {
                int iw;
                iw = iw0 - 0;
                if (iw >= 0 && iw < input_size) {
                    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    w_acc[3] += out_g * input[in_idx];
                }
                iw = iw0 - 1;
                if (iw >= 0 && iw < input_size) {
                    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    w_acc[4] += out_g * input[in_idx];
                }
                iw = iw0 - 2;
                if (iw >= 0 && iw < input_size) {
                    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    w_acc[5] += out_g * input[in_idx];
                }
            }
        }
        
        {
            int ih = ih0 - 2;
            if (ih >= 0 && ih < input_size) {
                int iw;
                iw = iw0 - 0;
                if (iw >= 0 && iw < input_size) {
                    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    w_acc[6] += out_g * input[in_idx];
                }
                iw = iw0 - 1;
                if (iw >= 0 && iw < input_size) {
                    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    w_acc[7] += out_g * input[in_idx];
                }
                iw = iw0 - 2;
                if (iw >= 0 && iw < input_size) {
                    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    w_acc[8] += out_g * input[in_idx];
                }
            }
        }

        if (ic == 0) {
            b_acc += out_g;
        }
    }

    
    extern __shared__ float s[];
    float* s_w = s;
    float* s_b = s + kk * blockDim.x;

#pragma unroll
    for (int k = 0; k < 9; k++) {
        s_w[k * blockDim.x + threadIdx.x] = w_acc[k];
    }
    s_b[threadIdx.x] = (ic == 0) ? b_acc : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (int k = 0; k < 9; k++) {
                s_w[k * blockDim.x + threadIdx.x] += s_w[k * blockDim.x + threadIdx.x + stride];
            }
            s_b[threadIdx.x] += s_b[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int kk_local = kk;
        int base = oc * (input_channels * kk_local) + ic * kk_local;
#pragma unroll
        for (int k = 0; k < 9; k++) {
            weight_gradients[base + k] = s_w[k * blockDim.x];
        }
        if (ic == 0) {
            bias_gradients[oc] = s_b[0];
        }
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

    
    
    extern __shared__ float s_w[];
    int kk = kernel_size * kernel_size;
    int w_count = output_channels * kk;
    for (int t = threadIdx.y * blockDim.x + threadIdx.x; t < w_count; t += blockDim.x * blockDim.y) {
        int oc = t / kk;
        int rem = t - oc * kk;
        int kh = rem / kernel_size;
        int kw = rem - kh * kernel_size;
        int w_idx = oc * (input_channels * kk) + ic * kk + kh * kernel_size + kw;
        s_w[t] = weights[w_idx];
    }
    __syncthreads();

    float in_sum = 0.0f;
    for (int oc = 0; oc < output_channels; oc++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                
                int oh = ih - padding + kh;
                int ow = iw - padding + kw;
                if (oh >= 0 && oh < output_size && ow >= 0 && ow < output_size) {
                    int out_idx = ((b * output_channels + oc) * output_size + oh) * output_size + ow;
                    in_sum += output_gradient[out_idx] * s_w[oc * kk + kh * kernel_size + kw];
                }
            }
        }
    }

    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
    input_gradients[in_idx] = in_sum;
}

void transpose_conv_backward(TransposeConvLayer* layer, float* d_input, float* d_output_gradient, int batch_size) {
    
    
    int threads = 256;
    int ocic_blocks = layer->output_channels * layer->input_channels;
    int kk = layer->kernel_size * layer->kernel_size;
    
    size_t shmem_wb = (size_t)(kk + 1) * (size_t)threads * sizeof(float);
    transpose_conv_weight_bias_grad_ocic_reduce_kernel<<<ocic_blocks, threads, shmem_wb>>>(
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
    size_t shmem_ig = (size_t)layer->output_channels * (size_t)kk * sizeof(float);
    transpose_conv_input_grad_kernel<<<ig_grid, ig_block, shmem_ig>>>(
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
    
    float* saved_out = decoder->tconv2->d_output;
    decoder->tconv2->d_output = decoder->d_reconstructed;
    transpose_conv_forward(decoder->tconv2, decoder->upsample2->d_output, batch_size);
    decoder->tconv2->d_output = saved_out;
}

void decoder_backward(Decoder* decoder, float* d_output_gradient, float* d_input, int batch_size) {
    
    

    transpose_conv_backward(decoder->tconv2, decoder->upsample2->d_output, d_output_gradient, batch_size);
    upsample_backward(decoder->upsample2, decoder->tconv2->d_input_gradients, batch_size);

    relu_backward(decoder->tconv1->d_output, decoder->upsample2->d_input_gradients,
                  decoder->d_tconv1_relu_grad,
                  batch_size * 64 * 16 * 16);

    transpose_conv_backward(decoder->tconv1, decoder->upsample1->d_output, decoder->d_tconv1_relu_grad, batch_size);
    upsample_backward(decoder->upsample1, decoder->tconv1->d_input_gradients, batch_size);

    
    if (d_input != decoder->upsample1->d_input_gradients) {
        CUDA_CHECK(cudaMemcpy(d_input, decoder->upsample1->d_input_gradients,
                              batch_size * 128 * 8 * 8 * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }
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
