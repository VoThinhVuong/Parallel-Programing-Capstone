#ifndef BACKWARD_H
#define BACKWARD_H

#include "cnn.h"

// Backward pass functions
void softmax_cross_entropy_backward(float* output, uint8_t* labels, float* gradient, int batch_size, int num_classes);
void fc_backward(FCLayer* layer, float* input, float* output_gradient, int batch_size);
void relu_backward(float* input, float* output_gradient, float* input_gradient, int size);
void maxpool_backward(MaxPoolLayer* layer, float* output_gradient, int batch_size);
void conv_backward(ConvLayer* layer, float* input, float* output_gradient, int batch_size);

// Complete backward pass through the network
void backward_pass(CNN* cnn, float* input, uint8_t* labels);

// Update weights using gradients (SGD)
void update_weights(CNN* cnn, float learning_rate);

#endif // BACKWARD_H
