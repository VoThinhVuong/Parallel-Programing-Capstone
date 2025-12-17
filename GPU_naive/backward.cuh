#ifndef BACKWARD_CUH
#define BACKWARD_CUH

#include "cnn.cuh"

// CUDA kernel declarations for backward pass
__global__ void softmax_cross_entropy_backward_kernel(float* output, uint8_t* labels, float* gradient,
                                                     int batch_size, int num_classes);

__global__ void fc_backward_kernel(float* input, float* weights, float* output_gradient,
                                  float* weight_gradients, float* bias_gradients, float* input_gradients,
                                  int batch_size, int input_size, int output_size);

__global__ void relu_backward_kernel(float* input, float* output_gradient, float* input_gradient, int size);

__global__ void maxpool_backward_kernel(float* output_gradient, int* max_indices, float* input_gradients,
                                       int batch_size, int channels, int input_size, int output_size);

__global__ void conv_backward_kernel(float* input, float* weights, float* output_gradient,
                                    float* weight_gradients, float* bias_gradients, float* input_gradients,
                                    int batch_size, int input_channels, int output_channels,
                                    int input_size, int output_size, int kernel_size,
                                    int stride, int padding);

__global__ void update_weights_kernel(float* weights, float* gradients, float learning_rate, int size);

// Host wrapper functions
void softmax_cross_entropy_backward(float* d_output, uint8_t* d_labels, float* d_gradient,
                                    int batch_size, int num_classes);
void fc_backward(FCLayer* layer, float* d_input, float* d_output_gradient, int batch_size);
void relu_backward(float* d_input, float* d_output_gradient, float* d_input_gradient, int size);
void maxpool_backward(MaxPoolLayer* layer, float* d_output_gradient, int batch_size);
void conv_backward(ConvLayer* layer, float* d_input, float* d_output_gradient, int batch_size);

// Complete backward pass through the network
void backward_pass(CNN* cnn, float* d_input, uint8_t* d_labels);

// Update weights using gradients (SGD)
void update_weights(CNN* cnn, float learning_rate);
void update_classifier_weights(CNN* cnn, float learning_rate);
void update_encoder_weights(CNN* cnn, float learning_rate);

// Decoder backward pass
__global__ void transpose_conv_backward_kernel(float* input, float* weights, float* output_gradient,
                                              float* weight_gradients, float* bias_gradients, float* input_gradients,
                                              int batch_size, int input_channels, int output_channels,
                                              int input_size, int output_size, int kernel_size,
                                              int stride, int padding);

__global__ void upsample_backward_kernel(float* output_gradient, float* input_gradients,
                                        int batch_size, int channels,
                                        int input_size, int output_size, int scale);

void transpose_conv_backward(TransposeConvLayer* layer, float* d_input, float* d_output_gradient, int batch_size);
void upsample_backward(UpsampleLayer* layer, float* d_output_gradient, int batch_size);
void decoder_backward(Decoder* decoder, float* d_output_gradient, float* d_input, int batch_size);
void update_decoder_weights(Decoder* decoder, float learning_rate);

// Backpropagate reconstruction gradients to encoder
void backprop_reconstruction_to_encoder(CNN* cnn, float* d_input, float* d_pool2_gradient, int batch_size);

#endif // BACKWARD_CUH
