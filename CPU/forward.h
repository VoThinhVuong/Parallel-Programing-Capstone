#ifndef FORWARD_H
#define FORWARD_H

#include "cnn.h"

// Forward pass functions
void conv_forward(ConvLayer* layer, float* input, int batch_size);
void relu_forward(float* input, float* output, int size);
void maxpool_forward(MaxPoolLayer* layer, float* input, int batch_size);
void fc_forward(FCLayer* layer, float* input, int batch_size);
void softmax_forward(float* input, float* output, int batch_size, int num_classes);

// Complete forward pass through the network
void forward_pass(CNN* cnn, float* input);

#endif // FORWARD_H
