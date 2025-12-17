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

// ------------------------------
// OPTIMIZED FC backward kernels
// ------------------------------

// Bias gradient: bias_grad[o] = sum_b out_grad[b,o]
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

// Weight gradient: w_grad[o,i] = sum_b out_grad[b,o] * input[b,i]
// One thread computes one (o,i) pair; reduction over batch happens in registers.
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

// Input gradient: in_grad[b,i] = sum_o out_grad[b,o] * weights[o,i]
// One block per (batch b), threads over input i.
__global__ void fc_input_grad_kernel(const float* __restrict__ weights,
                                    const float* __restrict__ output_gradient,
                                    float* __restrict__ input_gradients,
                                    int batch_size, int input_size, int output_size) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size || i >= input_size) return;

    float sum = 0.0f;
    // Iterate over output neurons
    for (int o = 0; o < output_size; o++) {
        sum += output_gradient[b * output_size + o] * weights[o * input_size + i];
    }
    input_gradients[b * input_size + i] = sum;
}

// ReLU backward kernel (unchanged - already atomic-free)
__global__ void relu_backward_kernel(float* input, float* output_gradient, float* input_gradient, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input_gradient[idx] = (input[idx] > 0) ? output_gradient[idx] : 0;
    }
}

// Max pooling backward kernel - ATOMIC-FREE using output-centric parallelization
// Each output position directly writes to its corresponding input position (no scatter conflicts)
__global__ void maxpool_backward_kernel(float* output_gradient, int* max_indices, float* input_gradients,
                                       int batch_size, int channels, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * output_size * output_size;
    
    if (idx >= total_outputs) {
        return;
    }
    
    // Direct write - each output gradient goes to exactly one input position
    int max_idx = max_indices[idx];
    input_gradients[max_idx] = output_gradient[idx];
}

// Convolution backward kernel - ATOMIC-FREE VERSION
// Strategy 1: Weight gradients - one thread per weight element, accumulate over batch/spatial
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
    
    // Accumulate over batch and spatial dimensions
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
                    
                    // Only first kernel position accumulates bias
                    if (ic == 0 && kh == 0 && kw == 0) {
                        bias_grad += output_gradient[output_idx];
                    }
                }
            }
        }
    }
    
    weight_gradients[weight_idx] = weight_grad;
    
    // Only one thread writes bias
    if (ic == 0 && kh == 0 && kw == 0) {
        bias_gradients[oc] = bias_grad;
    }
}

// Same as conv_backward_weights_kernel, but accumulates into existing weight/bias gradients.
// This is used to add reconstruction gradients on top of classification gradients.
__global__ void conv_backward_weights_accumulate_kernel(float* input, float* output_gradient,
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

    weight_gradients[weight_idx] += weight_grad;

    if (ic == 0 && kh == 0 && kw == 0) {
        bias_gradients[oc] += bias_grad;
    }
}

// Strategy 2: Input gradients - one thread per input element, accumulate over filters/kernel
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
    
    // Accumulate contributions from all output positions this input affects
    for (int oc = 0; oc < output_channels; oc++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // Find output position that this input contributes to
                int oh = (ih + padding - kh) / stride;
                int ow = (iw + padding - kw) / stride;
                
                // Check if this is a valid output position
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

// Weight update kernel (unchanged - already atomic-free)
__global__ void update_weights_kernel(float* weights, float* gradients, float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}

// Clear gradient kernel (unchanged)
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

// ------------------------------
// OPTIMIZED CONV backward kernels
// ------------------------------

// Weight + bias gradients (atomic-free) for common case: stride=1, kernel=3.
// One block computes one (oc, ic) pair and produces all 3x3 weight grads.
// Greatly reduces per-thread work versus 9 threads each doing the full reduction.
__global__ void conv_wb_grad_ocic_reduce_k3s1_kernel(const float* __restrict__ input,
                                                    const float* __restrict__ output_gradient,
                                                    float* __restrict__ weight_gradients,
                                                    float* __restrict__ bias_gradients,
                                                    int batch_size,
                                                    int input_channels,
                                                    int output_channels,
                                                    int input_size,
                                                    int output_size,
                                                    int padding) {
    int ocic = blockIdx.x;
    int oc = ocic / input_channels;
    int ic = ocic - oc * input_channels;
    if (oc >= output_channels || ic >= input_channels) return;

    // Accumulators for 3x3 weights
    float w_acc[9];
#pragma unroll
    for (int k = 0; k < 9; k++) w_acc[k] = 0.0f;
    float b_acc = 0.0f;

    int n = batch_size * output_size * output_size;
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        int ow = t % output_size;
        int tmp = t / output_size;
        int oh = tmp % output_size;
        int b = tmp / output_size;

        int out_idx = ((b * output_channels + oc) * output_size + oh) * output_size + ow;
        float out_g = output_gradient[out_idx];

        int ih0 = oh - padding;
        int iw0 = ow - padding;

        // kh=0
        {
            int ih = ih0 + 0;
            if (ih >= 0 && ih < input_size) {
                int iw;
                iw = iw0 + 0;
                if (iw >= 0 && iw < input_size) {
                    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    w_acc[0] += out_g * input[in_idx];
                }
                iw = iw0 + 1;
                if (iw >= 0 && iw < input_size) {
                    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    w_acc[1] += out_g * input[in_idx];
                }
                iw = iw0 + 2;
                if (iw >= 0 && iw < input_size) {
                    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    w_acc[2] += out_g * input[in_idx];
                }
            }
        }

        // kh=1
        {
            int ih = ih0 + 1;
            if (ih >= 0 && ih < input_size) {
                int iw;
                iw = iw0 + 0;
                if (iw >= 0 && iw < input_size) {
                    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    w_acc[3] += out_g * input[in_idx];
                }
                iw = iw0 + 1;
                if (iw >= 0 && iw < input_size) {
                    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    w_acc[4] += out_g * input[in_idx];
                }
                iw = iw0 + 2;
                if (iw >= 0 && iw < input_size) {
                    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    w_acc[5] += out_g * input[in_idx];
                }
            }
        }

        // kh=2
        {
            int ih = ih0 + 2;
            if (ih >= 0 && ih < input_size) {
                int iw;
                iw = iw0 + 0;
                if (iw >= 0 && iw < input_size) {
                    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    w_acc[6] += out_g * input[in_idx];
                }
                iw = iw0 + 1;
                if (iw >= 0 && iw < input_size) {
                    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    w_acc[7] += out_g * input[in_idx];
                }
                iw = iw0 + 2;
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

    // Shared memory layout: [9][blockDim.x] + [1][blockDim.x] for bias
    extern __shared__ float s[];
    float* s_w = s;
    float* s_b = s + 9 * blockDim.x;

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
        int base = oc * (input_channels * 9) + ic * 9;
#pragma unroll
        for (int k = 0; k < 9; k++) {
            weight_gradients[base + k] = s_w[k * blockDim.x];
        }
        if (ic == 0) {
            bias_gradients[oc] = s_b[0];
        }
    }
}

// Accumulate version: adds into existing weight/bias gradients.
__global__ void conv_wb_grad_ocic_accumulate_k3s1_kernel(const float* __restrict__ input,
                                                        const float* __restrict__ output_gradient,
                                                        float* __restrict__ weight_gradients,
                                                        float* __restrict__ bias_gradients,
                                                        int batch_size,
                                                        int input_channels,
                                                        int output_channels,
                                                        int input_size,
                                                        int output_size,
                                                        int padding) {
    int ocic = blockIdx.x;
    int oc = ocic / input_channels;
    int ic = ocic - oc * input_channels;
    if (oc >= output_channels || ic >= input_channels) return;

    float w_acc[9];
#pragma unroll
    for (int k = 0; k < 9; k++) w_acc[k] = 0.0f;
    float b_acc = 0.0f;

    int n = batch_size * output_size * output_size;
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        int ow = t % output_size;
        int tmp = t / output_size;
        int oh = tmp % output_size;
        int b = tmp / output_size;

        int out_idx = ((b * output_channels + oc) * output_size + oh) * output_size + ow;
        float out_g = output_gradient[out_idx];

        int ih0 = oh - padding;
        int iw0 = ow - padding;

        // kh=0
        {
            int ih = ih0 + 0;
            if (ih >= 0 && ih < input_size) {
                int iw;
                iw = iw0 + 0;
                if (iw >= 0 && iw < input_size) {
                    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    w_acc[0] += out_g * input[in_idx];
                }
                iw = iw0 + 1;
                if (iw >= 0 && iw < input_size) {
                    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    w_acc[1] += out_g * input[in_idx];
                }
                iw = iw0 + 2;
                if (iw >= 0 && iw < input_size) {
                    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    w_acc[2] += out_g * input[in_idx];
                }
            }
        }
        // kh=1
        {
            int ih = ih0 + 1;
            if (ih >= 0 && ih < input_size) {
                int iw;
                iw = iw0 + 0;
                if (iw >= 0 && iw < input_size) {
                    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    w_acc[3] += out_g * input[in_idx];
                }
                iw = iw0 + 1;
                if (iw >= 0 && iw < input_size) {
                    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    w_acc[4] += out_g * input[in_idx];
                }
                iw = iw0 + 2;
                if (iw >= 0 && iw < input_size) {
                    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    w_acc[5] += out_g * input[in_idx];
                }
            }
        }
        // kh=2
        {
            int ih = ih0 + 2;
            if (ih >= 0 && ih < input_size) {
                int iw;
                iw = iw0 + 0;
                if (iw >= 0 && iw < input_size) {
                    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    w_acc[6] += out_g * input[in_idx];
                }
                iw = iw0 + 1;
                if (iw >= 0 && iw < input_size) {
                    int in_idx = ((b * input_channels + ic) * input_size + ih) * input_size + iw;
                    w_acc[7] += out_g * input[in_idx];
                }
                iw = iw0 + 2;
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
    float* s_b = s + 9 * blockDim.x;

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
        int base = oc * (input_channels * 9) + ic * 9;
#pragma unroll
        for (int k = 0; k < 9; k++) {
            weight_gradients[base + k] += s_w[k * blockDim.x];
        }
        if (ic == 0) {
            bias_gradients[oc] += s_b[0];
        }
    }
}

// Input gradients for conv (stride=1) with weights cached per (ic) in shared memory.
__global__ void conv_input_grad_k3s1_cached_kernel(const float* __restrict__ output_gradient,
                                                  const float* __restrict__ weights,
                                                  float* __restrict__ input_gradients,
                                                  int batch_size, int input_channels, int output_channels,
                                                  int input_size, int output_size, int padding) {
    int b = blockIdx.z;
    int ic = blockIdx.y;

    int grid_w = (input_size + blockDim.y - 1) / blockDim.y;
    int ih_block = blockIdx.x / grid_w;
    int iw_block = blockIdx.x % grid_w;
    int ih = ih_block * blockDim.x + threadIdx.x;
    int iw = iw_block * blockDim.y + threadIdx.y;

    if (b >= batch_size || ic >= input_channels || ih >= input_size || iw >= input_size) return;

    // Cache weights for this (ic) across all oc and 3x3.
    extern __shared__ float s_w[];
    int kk = 9;
    int w_count = output_channels * kk;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int tstride = blockDim.x * blockDim.y;
    for (int t = tid; t < w_count; t += tstride) {
        int oc = t / kk;
        int rem = t - oc * kk;
        int kh = rem / 3;
        int kw = rem - kh * 3;
        int w_idx = oc * (input_channels * kk) + ic * kk + kh * 3 + kw;
        s_w[t] = weights[w_idx];
    }
    __syncthreads();

    float input_grad = 0.0f;
    // Accumulate contributions from all output positions this input affects (stride=1)
    for (int oc = 0; oc < output_channels; oc++) {
#pragma unroll
        for (int kh = 0; kh < 3; kh++) {
#pragma unroll
            for (int kw = 0; kw < 3; kw++) {
                int oh = ih + padding - kh;
                int ow = iw + padding - kw;
                if (oh >= 0 && oh < output_size && ow >= 0 && ow < output_size) {
                    int out_idx = b * (output_channels * output_size * output_size) +
                                  oc * (output_size * output_size) +
                                  oh * output_size + ow;
                    input_grad += output_gradient[out_idx] * s_w[oc * kk + kh * 3 + kw];
                }
            }
        }
    }

    int input_idx = b * (input_channels * input_size * input_size) +
                    ic * (input_size * input_size) +
                    ih * input_size + iw;
    input_gradients[input_idx] = input_grad;
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
    // Compute gradients (optimized, no atomics)
    // NOTE: No need to clear gradient buffers here; kernels below overwrite all elements.
    int threads = 256;
    // 1) bias gradients
    int bias_blocks = (layer->output_size + threads - 1) / threads;
    fc_bias_grad_kernel<<<bias_blocks, threads>>>(
        d_output_gradient, layer->d_bias_gradients, batch_size, layer->output_size);
    CUDA_CHECK(cudaGetLastError());

    // 2) weight gradients
    int wg_blocks_x = (layer->input_size + threads - 1) / threads;
    dim3 wg_grid(wg_blocks_x, layer->output_size);
    fc_weight_grad_kernel<<<wg_grid, threads>>>(
        d_input, d_output_gradient, layer->d_weight_gradients,
        batch_size, layer->input_size, layer->output_size);
    CUDA_CHECK(cudaGetLastError());

    // 3) input gradients
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
    // Clear input gradients first
    int input_total_size = batch_size * layer->input_channels * layer->input_size * layer->input_size;
    int threads = 256;
    clear_gradients_kernel<<<(input_total_size + threads - 1) / threads, threads>>>(
        layer->d_input_gradients, input_total_size);
    
    // Atomic-free backward pass - direct scatter
    int output_total_size = batch_size * layer->input_channels * layer->output_size * layer->output_size;
    maxpool_backward_kernel<<<(output_total_size + threads - 1) / threads, threads>>>(
        d_output_gradient, layer->d_max_indices, layer->d_input_gradients,
        batch_size, layer->input_channels, layer->output_size);
    CUDA_CHECK(cudaGetLastError());
}

void conv_backward(ConvLayer* layer, float* d_input, float* d_output_gradient, int batch_size) {
    // Step 1: Compute weight and bias gradients (atomic-free)
    if (layer->stride == 1 && layer->kernel_size == 3) {
        int threads = 256;
        int blocks = layer->output_channels * layer->input_channels;
        size_t shmem = (size_t)(9 + 1) * (size_t)threads * sizeof(float);
        conv_wb_grad_ocic_reduce_k3s1_kernel<<<blocks, threads, shmem>>>(
            d_input, d_output_gradient,
            layer->d_weight_gradients, layer->d_bias_gradients,
            batch_size,
            layer->input_channels,
            layer->output_channels,
            layer->input_size,
            layer->output_size,
            layer->padding);
        CUDA_CHECK(cudaGetLastError());
    } else {
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
    }

    // Step 2: Compute input gradients (atomic-free)
    if (layer->stride == 1 && layer->kernel_size == 3) {
        dim3 input_block(8, 8);
        int grid_h = (layer->input_size + input_block.x - 1) / input_block.x;
        int grid_w = (layer->input_size + input_block.y - 1) / input_block.y;
        dim3 input_grid(grid_h * grid_w, layer->input_channels, batch_size);
        size_t shmem = (size_t)layer->output_channels * (size_t)9 * sizeof(float);
        conv_input_grad_k3s1_cached_kernel<<<input_grid, input_block, shmem>>>(
            d_output_gradient, layer->d_weights,
            layer->d_input_gradients,
            batch_size, layer->input_channels, layer->output_channels,
            layer->input_size, layer->output_size,
            layer->padding);
        CUDA_CHECK(cudaGetLastError());
    } else {
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
}

// Backward pass for conv that ACCUMULATES weight/bias gradients into existing buffers.
// Input gradients are still overwritten (same as conv_backward).
static void conv_backward_accumulate(ConvLayer* layer, float* d_input, float* d_output_gradient, int batch_size) {
    if (layer->stride == 1 && layer->kernel_size == 3) {
        int threads = 256;
        int blocks = layer->output_channels * layer->input_channels;
        size_t shmem = (size_t)(9 + 1) * (size_t)threads * sizeof(float);
        conv_wb_grad_ocic_accumulate_k3s1_kernel<<<blocks, threads, shmem>>>(
            d_input, d_output_gradient,
            layer->d_weight_gradients, layer->d_bias_gradients,
            batch_size,
            layer->input_channels,
            layer->output_channels,
            layer->input_size,
            layer->output_size,
            layer->padding);
        CUDA_CHECK(cudaGetLastError());
    } else {
        dim3 weight_block(8, 8);
        dim3 weight_grid((layer->kernel_size + weight_block.x - 1) / weight_block.x,
                         layer->input_channels,
                         layer->output_channels);

        conv_backward_weights_accumulate_kernel<<<weight_grid, weight_block>>>(
            d_input, d_output_gradient,
            layer->d_weight_gradients, layer->d_bias_gradients,
            batch_size, layer->input_channels, layer->output_channels,
            layer->input_size, layer->output_size, layer->kernel_size,
            layer->stride, layer->padding);
        CUDA_CHECK(cudaGetLastError());
    }

    if (layer->stride == 1 && layer->kernel_size == 3) {
        dim3 input_block(8, 8);
        int grid_h = (layer->input_size + input_block.x - 1) / input_block.x;
        int grid_w = (layer->input_size + input_block.y - 1) / input_block.y;
        dim3 input_grid(grid_h * grid_w, layer->input_channels, batch_size);
        size_t shmem = (size_t)layer->output_channels * (size_t)9 * sizeof(float);
        conv_input_grad_k3s1_cached_kernel<<<input_grid, input_block, shmem>>>(
            d_output_gradient, layer->d_weights,
            layer->d_input_gradients,
            batch_size, layer->input_channels, layer->output_channels,
            layer->input_size, layer->output_size,
            layer->padding);
        CUDA_CHECK(cudaGetLastError());
    } else {
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
}

// Conv backward that only computes weight/bias gradients (no input gradients).
// Useful for conv1 where input gradients are never consumed.
static void conv_backward_weights_only(ConvLayer* layer, float* d_input, float* d_output_gradient, int batch_size) {
    if (layer->stride == 1 && layer->kernel_size == 3) {
        int threads = 256;
        int blocks = layer->output_channels * layer->input_channels;
        size_t shmem = (size_t)(9 + 1) * (size_t)threads * sizeof(float);
        conv_wb_grad_ocic_reduce_k3s1_kernel<<<blocks, threads, shmem>>>(
            d_input, d_output_gradient,
            layer->d_weight_gradients, layer->d_bias_gradients,
            batch_size,
            layer->input_channels,
            layer->output_channels,
            layer->input_size,
            layer->output_size,
            layer->padding);
        CUDA_CHECK(cudaGetLastError());
    } else {
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
    }
}

static void conv_backward_accumulate_weights_only(ConvLayer* layer, float* d_input, float* d_output_gradient, int batch_size) {
    if (layer->stride == 1 && layer->kernel_size == 3) {
        int threads = 256;
        int blocks = layer->output_channels * layer->input_channels;
        size_t shmem = (size_t)(9 + 1) * (size_t)threads * sizeof(float);
        conv_wb_grad_ocic_accumulate_k3s1_kernel<<<blocks, threads, shmem>>>(
            d_input, d_output_gradient,
            layer->d_weight_gradients, layer->d_bias_gradients,
            batch_size,
            layer->input_channels,
            layer->output_channels,
            layer->input_size,
            layer->output_size,
            layer->padding);
        CUDA_CHECK(cudaGetLastError());
    } else {
        dim3 weight_block(8, 8);
        dim3 weight_grid((layer->kernel_size + weight_block.x - 1) / weight_block.x,
                         layer->input_channels,
                         layer->output_channels);

        conv_backward_weights_accumulate_kernel<<<weight_grid, weight_block>>>(
            d_input, d_output_gradient,
            layer->d_weight_gradients, layer->d_bias_gradients,
            batch_size, layer->input_channels, layer->output_channels,
            layer->input_size, layer->output_size, layer->kernel_size,
            layer->stride, layer->padding);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Complete backward pass
void backward_pass(CNN* cnn, float* d_input, uint8_t* d_labels) {
    int batch_size = cnn->batch_size;

    // Reuse temporary gradient buffer (allocated once in create_cnn)
    float* d_fc2_grad = cnn->d_fc2_grad;
    
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
    // Conv1 input gradients are unused, so compute only weight/bias grads.
    conv_backward_weights_only(cnn->conv1, d_input, cnn->d_conv1_relu_grad, batch_size);
}

// Update all weights
void update_weights(CNN* cnn, float learning_rate) {
    update_encoder_weights(cnn, learning_rate);
    update_classifier_weights(cnn, learning_rate);
}

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

void backprop_reconstruction_to_encoder(CNN* cnn, float* d_input, float* d_pool2_gradient, int batch_size) {
    // Backprop from pool2 output (same shape as pool2->d_output) into conv layers.
    // We must accumulate conv weight/bias gradients on top of existing classification gradients.

    // Pool2 backward
    maxpool_backward(cnn->pool2, d_pool2_gradient, batch_size);

    // ReLU backward (Conv2)
    relu_backward(cnn->conv2->d_output, cnn->pool2->d_input_gradients, cnn->d_conv2_relu_grad,
                  batch_size * CONV2_FILTERS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE);

    // Conv2 backward: accumulate weight/bias gradients into existing classification gradients
    conv_backward_accumulate(cnn->conv2, cnn->pool1->d_output, cnn->d_conv2_relu_grad, batch_size);

    // Pool1 backward
    maxpool_backward(cnn->pool1, cnn->conv2->d_input_gradients, batch_size);

    // ReLU backward (Conv1)
    relu_backward(cnn->conv1->d_output, cnn->pool1->d_input_gradients, cnn->d_conv1_relu_grad,
                  batch_size * CONV1_FILTERS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE);

    // Conv1 backward: accumulate weight/bias gradients into existing classification gradients
    // Conv1 input gradients are unused, so accumulate only weight/bias grads.
    conv_backward_accumulate_weights_only(cnn->conv1, d_input, cnn->d_conv1_relu_grad, batch_size);
}
