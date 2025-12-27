#include "forward.cuh"
#include "cnn.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <math.h>


__global__ void conv_forward_kernel(float* input, float* weights, float* bias, float* output,
                                   int batch_size, int input_channels, int output_channels,
                                   int input_size, int output_size, int kernel_size,
                                   int stride, int padding) {
    int b = blockIdx.z;  
    int oc = blockIdx.y;  
    int oh = blockIdx.x * blockDim.x + threadIdx.x;  
    int ow = threadIdx.y;  
    
    if (b >= batch_size || oc >= output_channels || oh >= output_size || ow >= output_size) {
        return;
    }
    
    float sum = bias[oc];
    
    
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
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }
    
    int output_idx = b * (output_channels * output_size * output_size) +
                    oc * (output_size * output_size) +
                    oh * output_size + ow;
    output[output_idx] = sum;
}


__global__ void relu_forward_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input[idx] > 0) ? input[idx] : 0;
    }
}


__global__ void maxpool_forward_kernel(float* input, float* output, int* max_indices,
                                      int batch_size, int channels, int input_size,
                                      int output_size, int pool_size, int stride) {
    int b = blockIdx.z;  
    int c = blockIdx.y;  
    int oh = blockIdx.x * blockDim.x + threadIdx.x;  
    int ow = threadIdx.y;  
    
    if (b >= batch_size || c >= channels || oh >= output_size || ow >= output_size) {
        return;
    }
    
    float max_val = -FLT_MAX;
    int max_idx = 0;
    
    
    for (int ph = 0; ph < pool_size; ph++) {
        for (int pw = 0; pw < pool_size; pw++) {
            int ih = oh * stride + ph;
            int iw = ow * stride + pw;
            
            int input_idx = b * (channels * input_size * input_size) +
                          c * (input_size * input_size) +
                          ih * input_size + iw;
            
            if (input[input_idx] > max_val) {
                max_val = input[input_idx];
                max_idx = input_idx;
            }
        }
    }
    
    int output_idx = b * (channels * output_size * output_size) +
                    c * (output_size * output_size) +
                    oh * output_size + ow;
    output[output_idx] = max_val;
    max_indices[output_idx] = max_idx;
}


__global__ void fc_forward_kernel(float* input, float* weights, float* bias, float* output,
                                 int batch_size, int input_size, int output_size) {
    int b = blockIdx.y;  
    int o = blockIdx.x * blockDim.x + threadIdx.x;  
    
    if (b >= batch_size || o >= output_size) {
        return;
    }
    
    float sum = bias[o];
    for (int i = 0; i < input_size; i++) {
        sum += input[b * input_size + i] * weights[o * input_size + i];
    }
    
    output[b * output_size + o] = sum;
}


__global__ void softmax_forward_kernel(float* input, float* output, int batch_size, int num_classes) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= batch_size) {
        return;
    }
    
    
    float max_val = input[b * num_classes];
    for (int i = 1; i < num_classes; i++) {
        if (input[b * num_classes + i] > max_val) {
            max_val = input[b * num_classes + i];
        }
    }
    
    
    float sum = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        output[b * num_classes + i] = expf(input[b * num_classes + i] - max_val);
        sum += output[b * num_classes + i];
    }
    
    
    for (int i = 0; i < num_classes; i++) {
        output[b * num_classes + i] /= sum;
    }
}


void conv_forward(ConvLayer* layer, float* d_input, int batch_size) {
    dim3 block(8, 8);
    dim3 grid((layer->output_size + block.x - 1) / block.x, layer->output_channels, batch_size);
    
    conv_forward_kernel<<<grid, block>>>(d_input, layer->d_weights, layer->d_bias, layer->d_output,
                                        batch_size, layer->input_channels, layer->output_channels,
                                        layer->input_size, layer->output_size, layer->kernel_size,
                                        layer->stride, layer->padding);
    CUDA_CHECK(cudaGetLastError());
}

void relu_forward(float* d_input, float* d_output, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    relu_forward_kernel<<<blocks, threads>>>(d_input, d_output, size);
    CUDA_CHECK(cudaGetLastError());
}

void maxpool_forward(MaxPoolLayer* layer, float* d_input, int batch_size) {
    dim3 block(8, 8);
    dim3 grid((layer->output_size + block.x - 1) / block.x, layer->input_channels, batch_size);
    
    maxpool_forward_kernel<<<grid, block>>>(d_input, layer->d_output, layer->d_max_indices,
                                           batch_size, layer->input_channels,
                                           layer->input_size, layer->output_size,
                                           layer->pool_size, layer->stride);
    CUDA_CHECK(cudaGetLastError());
}

void fc_forward(FCLayer* layer, float* d_input, int batch_size) {
    int threads = 256;
    int blocks = (layer->output_size + threads - 1) / threads;
    dim3 grid(blocks, batch_size);
    
    fc_forward_kernel<<<grid, threads>>>(d_input, layer->d_weights, layer->d_bias, layer->d_output,
                                        batch_size, layer->input_size, layer->output_size);
    CUDA_CHECK(cudaGetLastError());
}

void softmax_forward(float* d_input, float* d_output, int batch_size, int num_classes) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    
    softmax_forward_kernel<<<blocks, threads>>>(d_input, d_output, batch_size, num_classes);
    CUDA_CHECK(cudaGetLastError());
}


void forward_pass(CNN* cnn, float* d_input) {
    int batch_size = cnn->batch_size;
    
    
    conv_forward(cnn->conv1, d_input, batch_size);
    relu_forward(cnn->conv1->d_output, cnn->d_conv1_relu,
                 batch_size * CONV1_FILTERS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE);
    
    
    maxpool_forward(cnn->pool1, cnn->d_conv1_relu, batch_size);
    
    
    conv_forward(cnn->conv2, cnn->pool1->d_output, batch_size);
    relu_forward(cnn->conv2->d_output, cnn->d_conv2_relu,
                 batch_size * CONV2_FILTERS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE);
    
    
    maxpool_forward(cnn->pool2, cnn->d_conv2_relu, batch_size);
    
    
    fc_forward(cnn->fc1, cnn->pool2->d_output, batch_size);
    relu_forward(cnn->fc1->d_output, cnn->d_fc1_relu, batch_size * FC1_OUTPUT_SIZE);
    
    
    fc_forward(cnn->fc2, cnn->d_fc1_relu, batch_size);
    
    
    softmax_forward(cnn->fc2->d_output, cnn->d_output, batch_size, FC2_OUTPUT_SIZE);
}
