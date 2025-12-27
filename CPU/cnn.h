#ifndef CNN_H
#define CNN_H

#include <stdint.h>


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


typedef struct {
    int input_channels;
    int output_channels;
    int kernel_size;
    int stride;
    int padding;
    int input_size;
    int output_size;
    
    float* weights;  
    float* bias;     
    float* output;   
    
    
    float* weight_gradients;
    float* bias_gradients;
    float* input_gradients;
} ConvLayer;


typedef struct {
    int pool_size;
    int stride;
    int input_channels;
    int input_size;
    int output_size;
    
    float* output;   
    int* max_indices;  
    float* input_gradients;
} MaxPoolLayer;


typedef struct {
    int input_size;
    int output_size;
    
    float* weights;  
    float* bias;     
    float* output;   
    
    
    float* weight_gradients;
    float* bias_gradients;
    float* input_gradients;
} FCLayer;


typedef struct {
    int input_channels;
    int output_channels;
    int kernel_size;
    int stride;
    int padding;
    int input_size;
    int output_size;
    
    float* weights;  
    float* bias;     
    float* output;   
    
    
    float* weight_gradients;
    float* bias_gradients;
    float* input_gradients;
} TransposeConvLayer;


typedef struct {
    int channels;
    int input_size;
    int output_size;
    int scale_factor;
    
    float* output;   
    float* input_gradients;
} UpsampleLayer;


typedef struct {
    UpsampleLayer* upsample1;      
    TransposeConvLayer* tconv1;    
    UpsampleLayer* upsample2;      
    TransposeConvLayer* tconv2;    
    
    float* tconv1_relu;            
    float* tconv1_relu_grad;       
    float* reconstructed;          
    
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
    
    
    float* conv1_relu;  
    float* conv2_relu;  
    float* fc1_relu;    
    
    
    float* conv1_relu_grad;
    float* conv2_relu_grad;
    float* fc1_relu_grad;
    
    
    float* output;  
    
    int batch_size;
} CNN;


ConvLayer* create_conv_layer(int input_channels, int output_channels, 
                             int kernel_size, int stride, int padding,
                             int input_size, int batch_size);
MaxPoolLayer* create_maxpool_layer(int pool_size, int stride, 
                                   int input_channels, int input_size, 
                                   int batch_size);
FCLayer* create_fc_layer(int input_size, int output_size, int batch_size);
TransposeConvLayer* create_transpose_conv_layer(int input_channels, int output_channels,
                                                int kernel_size, int stride, int padding,
                                                int input_size, int batch_size);
UpsampleLayer* create_upsample_layer(int channels, int input_size, 
                                     int scale_factor, int batch_size);


void free_conv_layer(ConvLayer* layer);
void free_maxpool_layer(MaxPoolLayer* layer);
void free_fc_layer(FCLayer* layer);
void free_transpose_conv_layer(TransposeConvLayer* layer);
void free_upsample_layer(UpsampleLayer* layer);


Decoder* create_decoder(int batch_size);
void free_decoder(Decoder* decoder);
void initialize_decoder_weights(Decoder* decoder);


CNN* create_cnn(int batch_size);
void free_cnn(CNN* cnn);


void initialize_weights(CNN* cnn);

#endif 
