#ifndef CNN_CUH
#define CNN_CUH

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


#define INPUT_WIDTH 32
#define INPUT_HEIGHT 32
#define INPUT_CHANNELS 3


#define CONV1_FILTERS 32
#define CONV1_KERNEL_SIZE 3
#define CONV1_STRIDE 1
#define CONV1_PADDING 1
#define CONV1_OUTPUT_SIZE 32  


#define POOL1_SIZE 2
#define POOL1_STRIDE 2
#define POOL1_OUTPUT_SIZE 16  


#define CONV2_FILTERS 128
#define CONV2_KERNEL_SIZE 3
#define CONV2_STRIDE 1
#define CONV2_PADDING 1
#define CONV2_OUTPUT_SIZE 16  


#define POOL2_SIZE 2
#define POOL2_STRIDE 2
#define POOL2_OUTPUT_SIZE 8  


#define FC1_INPUT_SIZE (CONV2_FILTERS * POOL2_OUTPUT_SIZE * POOL2_OUTPUT_SIZE)  
#define FC1_OUTPUT_SIZE 128
#define FC2_INPUT_SIZE 128
#define FC2_OUTPUT_SIZE 10  


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


typedef struct {
    int input_channels;
    int output_channels;
    int kernel_size;
    int stride;
    int padding;
    int input_size;
    int output_size;
    
    
    float* d_weights;  
    float* d_bias;     
    float* d_output;   
    
    
    float* d_weight_gradients;
    float* d_bias_gradients;
    float* d_input_gradients;
} ConvLayer;


typedef struct {
    int pool_size;
    int stride;
    int input_channels;
    int input_size;
    int output_size;
    
    
    float* d_output;   
    int* d_max_indices;  
    float* d_input_gradients;
} MaxPoolLayer;


typedef struct {
    int input_size;
    int output_size;
    
    
    float* d_weights;  
    float* d_bias;     
    float* d_output;   
    
    
    float* d_weight_gradients;
    float* d_bias_gradients;
    float* d_input_gradients;
} FCLayer;


typedef struct {
    int input_channels;
    int output_channels;
    int kernel_size;
    int stride;
    int padding;
    int input_size;
    int output_size;

    
    float* d_weights;  
    float* d_bias;     
    float* d_output;   

    
    float* d_weight_gradients;
    float* d_bias_gradients;
    float* d_input_gradients;
} TransposeConvLayer;


typedef struct {
    int channels;
    int input_size;
    int output_size;
    int scale_factor;

    
    float* d_output;   
    float* d_input_gradients;
} UpsampleLayer;


typedef struct {
    UpsampleLayer* upsample1;      
    TransposeConvLayer* tconv1;    
    UpsampleLayer* upsample2;      
    TransposeConvLayer* tconv2;    

    
    float* d_tconv1_relu;          
    float* d_tconv1_relu_grad;     
    float* d_reconstructed;        

    int batch_size;
} Decoder;


typedef struct {
    
    ConvLayer* conv1;
    MaxPoolLayer* pool1;
    ConvLayer* conv2;
    MaxPoolLayer* pool2;
    FCLayer* fc1;
    FCLayer* fc2;

    
    Decoder* decoder;
    
    
    float* d_conv1_relu;  
    float* d_conv2_relu;  
    float* d_fc1_relu;    
    
    
    float* d_conv1_relu_grad;
    float* d_conv2_relu_grad;
    float* d_fc1_relu_grad;
    
    
    float* d_output;  

    
    float* d_fc2_grad; 
    
    int batch_size;
} CNN;


ConvLayer* create_conv_layer(int input_channels, int output_channels, 
                             int kernel_size, int stride, int padding,
                             int input_size, int batch_size);
MaxPoolLayer* create_maxpool_layer(int pool_size, int stride, 
                                   int input_channels, int input_size, 
                                   int batch_size);
FCLayer* create_fc_layer(int input_size, int output_size, int batch_size);


void free_conv_layer(ConvLayer* layer);
void free_maxpool_layer(MaxPoolLayer* layer);
void free_fc_layer(FCLayer* layer);

void free_transpose_conv_layer(TransposeConvLayer* layer);
void free_upsample_layer(UpsampleLayer* layer);
void free_decoder(Decoder* decoder);


TransposeConvLayer* create_transpose_conv_layer(int input_channels, int output_channels,
                                                int kernel_size, int stride, int padding,
                                                int input_size, int batch_size);
UpsampleLayer* create_upsample_layer(int channels, int input_size, int scale_factor, int batch_size);
Decoder* create_decoder(int batch_size);


CNN* create_cnn(int batch_size);
void free_cnn(CNN* cnn);


void initialize_weights(CNN* cnn);
void initialize_decoder_weights(Decoder* decoder);

#endif 
