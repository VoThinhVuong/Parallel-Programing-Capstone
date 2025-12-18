#ifndef CNN_CUH
#define CNN_CUH

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Network architecture parameters (same as CPU)
#define INPUT_WIDTH 32
#define INPUT_HEIGHT 32
#define INPUT_CHANNELS 3

// Conv1 layer: 32 filters, 3x3 kernel, stride 1, padding 1
#define CONV1_FILTERS 128
#define CONV1_KERNEL_SIZE 3
#define CONV1_STRIDE 1
#define CONV1_PADDING 1
#define CONV1_OUTPUT_SIZE 32  // (32 + 2*1 - 3)/1 + 1 = 32

// Pool1 layer: 2x2, stride 2
#define POOL1_SIZE 2
#define POOL1_STRIDE 2
#define POOL1_OUTPUT_SIZE 16  // 32/2 = 16

// Conv2 layer: 128 filters, 3x3 kernel, stride 1, padding 1
#define CONV2_FILTERS 128
#define CONV2_KERNEL_SIZE 3
#define CONV2_STRIDE 1
#define CONV2_PADDING 1
#define CONV2_OUTPUT_SIZE 16  // (16 + 2*1 - 3)/1 + 1 = 16

// Pool2 layer: 2x2, stride 2
#define POOL2_SIZE 2
#define POOL2_STRIDE 2
#define POOL2_OUTPUT_SIZE 8  // 16/2 = 8

// Fully connected layers
#define FC1_INPUT_SIZE (CONV2_FILTERS * POOL2_OUTPUT_SIZE * POOL2_OUTPUT_SIZE)  // 128*8*8 = 8192
#define FC1_OUTPUT_SIZE 128
#define FC2_INPUT_SIZE 128
#define FC2_OUTPUT_SIZE 10  // Number of classes

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Convolutional layer (GPU version)
typedef struct {
    int input_channels;
    int output_channels;
    int kernel_size;
    int stride;
    int padding;
    int input_size;
    int output_size;
    
    // Device pointers
    float* d_weights;  // output_channels x input_channels x kernel_size x kernel_size
    float* d_bias;     // output_channels
    float* d_output;   // batch_size x output_channels x output_size x output_size
    
    // For backpropagation
    float* d_weight_gradients;
    float* d_bias_gradients;
    float* d_input_gradients;
} ConvLayer;

// Max pooling layer (GPU version)
typedef struct {
    int pool_size;
    int stride;
    int input_channels;
    int input_size;
    int output_size;
    
    // Device pointers
    float* d_output;   // batch_size x channels x output_size x output_size
    int* d_max_indices;  // Store indices for backprop
    float* d_input_gradients;
} MaxPoolLayer;

// Fully connected layer (GPU version)
typedef struct {
    int input_size;
    int output_size;
    
    // Device pointers
    float* d_weights;  // output_size x input_size
    float* d_bias;     // output_size
    float* d_output;   // batch_size x output_size
    
    // For backpropagation
    float* d_weight_gradients;
    float* d_bias_gradients;
    float* d_input_gradients;
} FCLayer;

// Transpose convolution layer (for decoder)
typedef struct {
    int input_channels;
    int output_channels;
    int kernel_size;
    int stride;
    int padding;
    int input_size;
    int output_size;

    // Device pointers
    float* d_weights;  // output_channels x input_channels x kernel_size x kernel_size
    float* d_bias;     // output_channels
    float* d_output;   // batch_size x output_channels x output_size x output_size

    // For backpropagation
    float* d_weight_gradients;
    float* d_bias_gradients;
    float* d_input_gradients;
} TransposeConvLayer;

// Upsampling layer (nearest neighbor)
typedef struct {
    int channels;
    int input_size;
    int output_size;
    int scale_factor;

    // Device pointers
    float* d_output;   // batch_size x channels x output_size x output_size
    float* d_input_gradients;
} UpsampleLayer;

// Decoder network
typedef struct {
    ConvLayer* conv1;              // 128ch → 128ch, 8×8
    UpsampleLayer* upsample1;      // 8×8 → 16×16
    ConvLayer* conv2;              // 128ch → 256ch, 16×16
    UpsampleLayer* upsample2;      // 16×16 → 32×32
    ConvLayer* conv3;              // 256ch → 3ch, 32×32

    // Device pointers
    float* d_conv1_relu;           // After Conv1 + ReLU
    float* d_conv1_relu_grad;      // Gradient
    float* d_conv2_relu;           // After Conv2 + ReLU
    float* d_conv2_relu_grad;      // Gradient
    float* d_reconstructed;        // Final 32×32×3 output

    int batch_size;
} Decoder;

// Complete CNN model (GPU version)
typedef struct {
    // Layers
    ConvLayer* conv1;
    MaxPoolLayer* pool1;
    ConvLayer* conv2;
    MaxPoolLayer* pool2;
    FCLayer* fc1;
    FCLayer* fc2;

    // Decoder (optional, for reconstruction)
    Decoder* decoder;
    
    // Intermediate activations (device pointers)
    float* d_conv1_relu;  // After ReLU
    float* d_conv2_relu;  // After ReLU
    float* d_fc1_relu;    // After ReLU
    
    // Gradients for activations (device pointers)
    float* d_conv1_relu_grad;
    float* d_conv2_relu_grad;
    float* d_fc1_relu_grad;
    
    // Final output (softmax probabilities, device pointer)
    float* d_output;  // batch_size x 10
    
    int batch_size;
} CNN;

// Layer initialization functions
ConvLayer* create_conv_layer(int input_channels, int output_channels, 
                             int kernel_size, int stride, int padding,
                             int input_size, int batch_size);
MaxPoolLayer* create_maxpool_layer(int pool_size, int stride, 
                                   int input_channels, int input_size, 
                                   int batch_size);
FCLayer* create_fc_layer(int input_size, int output_size, int batch_size);

// Layer cleanup functions
void free_conv_layer(ConvLayer* layer);
void free_maxpool_layer(MaxPoolLayer* layer);
void free_fc_layer(FCLayer* layer);

void free_transpose_conv_layer(TransposeConvLayer* layer);
void free_upsample_layer(UpsampleLayer* layer);
void free_decoder(Decoder* decoder);

// Decoder layer creation
TransposeConvLayer* create_transpose_conv_layer(int input_channels, int output_channels,
                                                int kernel_size, int stride, int padding,
                                                int input_size, int batch_size);
UpsampleLayer* create_upsample_layer(int channels, int input_size, int scale_factor, int batch_size);
Decoder* create_decoder(int batch_size);

// CNN creation and cleanup
CNN* create_cnn(int batch_size);
void free_cnn(CNN* cnn);

// Initialize weights with random values (on host then copy to device)
void initialize_weights(CNN* cnn);
void initialize_decoder_weights(Decoder* decoder);

#endif // CNN_CUH
