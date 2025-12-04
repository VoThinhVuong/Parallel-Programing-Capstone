#ifndef CNN_H
#define CNN_H

#include <stdint.h>

// Network architecture parameters
#define INPUT_WIDTH 32
#define INPUT_HEIGHT 32
#define INPUT_CHANNELS 3

// Conv1 layer: 32 filters, 3x3 kernel, stride 1, padding 1
#define CONV1_FILTERS 32
#define CONV1_KERNEL_SIZE 3
#define CONV1_STRIDE 1
#define CONV1_PADDING 1
#define CONV1_OUTPUT_SIZE 32  // (32 + 2*1 - 3)/1 + 1 = 32

// Pool1 layer: 2x2, stride 2
#define POOL1_SIZE 2
#define POOL1_STRIDE 2
#define POOL1_OUTPUT_SIZE 16  // 32/2 = 16

// Conv2 layer: 64 filters, 3x3 kernel, stride 1, padding 1
#define CONV2_FILTERS 64
#define CONV2_KERNEL_SIZE 3
#define CONV2_STRIDE 1
#define CONV2_PADDING 1
#define CONV2_OUTPUT_SIZE 16  // (16 + 2*1 - 3)/1 + 1 = 16

// Pool2 layer: 2x2, stride 2
#define POOL2_SIZE 2
#define POOL2_STRIDE 2
#define POOL2_OUTPUT_SIZE 8  // 16/2 = 8

// Fully connected layers
#define FC1_INPUT_SIZE (CONV2_FILTERS * POOL2_OUTPUT_SIZE * POOL2_OUTPUT_SIZE)  // 64*8*8 = 4096
#define FC1_OUTPUT_SIZE 128
#define FC2_INPUT_SIZE 128
#define FC2_OUTPUT_SIZE 10  // Number of classes

// Convolutional layer
typedef struct {
    int input_channels;
    int output_channels;
    int kernel_size;
    int stride;
    int padding;
    int input_size;
    int output_size;
    
    float* weights;  // output_channels x input_channels x kernel_size x kernel_size
    float* bias;     // output_channels
    float* output;   // batch_size x output_channels x output_size x output_size
    
    // For backpropagation
    float* weight_gradients;
    float* bias_gradients;
    float* input_gradients;
} ConvLayer;

// Max pooling layer
typedef struct {
    int pool_size;
    int stride;
    int input_channels;
    int input_size;
    int output_size;
    
    float* output;   // batch_size x channels x output_size x output_size
    int* max_indices;  // Store indices for backprop
    float* input_gradients;
} MaxPoolLayer;

// Fully connected layer
typedef struct {
    int input_size;
    int output_size;
    
    float* weights;  // output_size x input_size
    float* bias;     // output_size
    float* output;   // batch_size x output_size
    
    // For backpropagation
    float* weight_gradients;
    float* bias_gradients;
    float* input_gradients;
} FCLayer;

// Complete CNN model
typedef struct {
    // Layers
    ConvLayer* conv1;
    MaxPoolLayer* pool1;
    ConvLayer* conv2;
    MaxPoolLayer* pool2;
    FCLayer* fc1;
    FCLayer* fc2;
    
    // Intermediate activations
    float* conv1_relu;  // After ReLU
    float* conv2_relu;  // After ReLU
    float* fc1_relu;    // After ReLU
    
    // Gradients for activations
    float* conv1_relu_grad;
    float* conv2_relu_grad;
    float* fc1_relu_grad;
    
    // Final output (softmax probabilities)
    float* output;  // batch_size x 10
    
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

// CNN creation and cleanup
CNN* create_cnn(int batch_size);
void free_cnn(CNN* cnn);

// Initialize weights with random values
void initialize_weights(CNN* cnn);

#endif // CNN_H
