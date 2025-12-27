#ifndef FORWARD_H
#define FORWARD_H

#include "cnn.h"


void conv_forward(ConvLayer* layer, float* input, int batch_size);
void relu_forward(float* input, float* output, int size);
void maxpool_forward(MaxPoolLayer* layer, float* input, int batch_size);
void fc_forward(FCLayer* layer, float* input, int batch_size);
void softmax_forward(float* input, float* output, int batch_size, int num_classes);


void upsample_forward(UpsampleLayer* layer, float* input, int batch_size);
void transpose_conv_forward(TransposeConvLayer* layer, float* input, int batch_size);
void decoder_forward(Decoder* decoder, float* input);


void forward_pass(CNN* cnn, float* input);

#endif 
