#ifndef FORWARD_CUH
#define FORWARD_CUH

#include "cnn.cuh"

// CUDA kernel declarations for forward pass
__global__ void conv_forward_kernel(float* input, float* weights, float* bias, float* output,
                                   int batch_size, int input_channels, int output_channels,
                                   int input_size, int output_size, int kernel_size,
                                   int stride, int padding);

__global__ void relu_forward_kernel(float* input, float* output, int size);

__global__ void maxpool_forward_kernel(float* input, float* output, int* max_indices,
                                      int batch_size, int channels, int input_size,
                                      int output_size, int pool_size, int stride);

__global__ void fc_forward_kernel(float* input, float* weights, float* bias, float* output,
                                 int batch_size, int input_size, int output_size);

__global__ void softmax_forward_kernel(float* input, float* output, int batch_size, int num_classes);

// Host wrapper functions
void conv_forward(ConvLayer* layer, float* d_input, int batch_size);
void relu_forward(float* d_input, float* d_output, int size);
void maxpool_forward(MaxPoolLayer* layer, float* d_input, int batch_size);
void fc_forward(FCLayer* layer, float* d_input, int batch_size);
void softmax_forward(float* d_input, float* d_output, int batch_size, int num_classes);

// Complete forward pass through the network
void forward_pass(CNN* cnn, float* d_input);

// Decoder forward pass functions
__global__ void upsample_forward_kernel(float* input, float* output,
                                       int batch_size, int channels,
                                       int input_size, int output_size, int scale);

__global__ void transpose_conv_forward_kernel(float* input, float* weights, float* bias, float* output,
                                             int batch_size, int input_channels, int output_channels,
                                             int input_size, int output_size, int kernel_size,
                                             int stride, int padding);

void upsample_forward(UpsampleLayer* layer, float* d_input, int batch_size);
void transpose_conv_forward(TransposeConvLayer* layer, float* d_input, int batch_size);
void decoder_forward(Decoder* decoder, float* d_input, int batch_size);

#endif // FORWARD_CUH
