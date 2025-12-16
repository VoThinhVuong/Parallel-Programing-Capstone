#include "backward.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Softmax + Cross-Entropy backward (combined for efficiency)
void softmax_cross_entropy_backward(float* output, uint8_t* labels, float* gradient, int batch_size, int num_classes) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < num_classes; i++) {
            int idx = b * num_classes + i;
            gradient[idx] = output[idx];
            if (i == labels[b]) {
                gradient[idx] -= 1.0f;
            }
            gradient[idx] /= batch_size;  // Average over batch
        }
    }
}

// Fully connected backward pass
void fc_backward(FCLayer* layer, float* input, float* output_gradient, int batch_size) {
    int input_size = layer->input_size;
    int output_size = layer->output_size;
    
    // Clear gradients
    memset(layer->weight_gradients, 0, output_size * input_size * sizeof(float));
    memset(layer->bias_gradients, 0, output_size * sizeof(float));
    memset(layer->input_gradients, 0, batch_size * input_size * sizeof(float));
    
    // Compute gradients
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < output_size; o++) {
            float out_grad = output_gradient[b * output_size + o];
            
            // Bias gradient
            layer->bias_gradients[o] += out_grad;
            
            // Weight and input gradients
            for (int i = 0; i < input_size; i++) {
                layer->weight_gradients[o * input_size + i] += out_grad * input[b * input_size + i];
                layer->input_gradients[b * input_size + i] += out_grad * layer->weights[o * input_size + i];
            }
        }
    }
}

// ReLU backward pass
void relu_backward(float* input, float* output_gradient, float* input_gradient, int size) {
    for (int i = 0; i < size; i++) {
        input_gradient[i] = (input[i] > 0) ? output_gradient[i] : 0;
    }
}

// Max pooling backward pass
void maxpool_backward(MaxPoolLayer* layer, float* output_gradient, int batch_size) {
    int output_size = layer->output_size;
    int channels = layer->input_channels;
    int input_size = layer->input_size;
    
    // Clear input gradients
    memset(layer->input_gradients, 0, batch_size * channels * input_size * input_size * sizeof(float));
    
    // Route gradients to max positions
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < output_size; oh++) {
                for (int ow = 0; ow < output_size; ow++) {
                    int output_idx = b * (channels * output_size * output_size) +
                                    c * (output_size * output_size) +
                                    oh * output_size + ow;
                    int max_idx = layer->max_indices[output_idx];
                    layer->input_gradients[max_idx] += output_gradient[output_idx];
                }
            }
        }
    }
}

// Convolution backward pass
void conv_backward(ConvLayer* layer, float* input, float* output_gradient, int batch_size) {
    int input_size = layer->input_size;
    int output_size = layer->output_size;
    int kernel_size = layer->kernel_size;
    int padding = layer->padding;
    int stride = layer->stride;
    int input_channels = layer->input_channels;
    int output_channels = layer->output_channels;
    
    // Clear gradients
    memset(layer->weight_gradients, 0, output_channels * input_channels * kernel_size * kernel_size * sizeof(float));
    memset(layer->bias_gradients, 0, output_channels * sizeof(float));
    memset(layer->input_gradients, 0, batch_size * input_channels * input_size * input_size * sizeof(float));
    
    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < output_channels; oc++) {
            for (int oh = 0; oh < output_size; oh++) {
                for (int ow = 0; ow < output_size; ow++) {
                    int output_idx = b * (output_channels * output_size * output_size) +
                                    oc * (output_size * output_size) +
                                    oh * output_size + ow;
                    float out_grad = output_gradient[output_idx];
                    
                    // Bias gradient
                    layer->bias_gradients[oc] += out_grad;
                    
                    // Weight and input gradients
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
                                    
                                    layer->weight_gradients[weight_idx] += out_grad * input[input_idx];
                                    layer->input_gradients[input_idx] += out_grad * layer->weights[weight_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Complete backward pass
void backward_pass(CNN* cnn, float* input, uint8_t* labels) {
    int batch_size = cnn->batch_size;
    
    // Softmax + Cross-Entropy gradient
    float* fc2_grad = (float*)malloc(batch_size * FC2_OUTPUT_SIZE * sizeof(float));
    softmax_cross_entropy_backward(cnn->output, labels, fc2_grad, batch_size, FC2_OUTPUT_SIZE);
    
    // FC2 backward
    fc_backward(cnn->fc2, cnn->fc1_relu, fc2_grad, batch_size);
    
    // ReLU backward (FC1)
    relu_backward(cnn->fc1->output, cnn->fc2->input_gradients, cnn->fc1_relu_grad, 
                  batch_size * FC1_OUTPUT_SIZE);
    
    // FC1 backward
    fc_backward(cnn->fc1, cnn->pool2->output, cnn->fc1_relu_grad, batch_size);
    
    // Pool2 backward
    maxpool_backward(cnn->pool2, cnn->fc1->input_gradients, batch_size);
    
    // ReLU backward (Conv2)
    relu_backward(cnn->conv2->output, cnn->pool2->input_gradients, cnn->conv2_relu_grad,
                  batch_size * CONV2_FILTERS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE);
    
    // Conv2 backward
    conv_backward(cnn->conv2, cnn->pool1->output, cnn->conv2_relu_grad, batch_size);
    
    // Pool1 backward
    maxpool_backward(cnn->pool1, cnn->conv2->input_gradients, batch_size);
    
    // ReLU backward (Conv1)
    relu_backward(cnn->conv1->output, cnn->pool1->input_gradients, cnn->conv1_relu_grad,
                  batch_size * CONV1_FILTERS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE);
    
    // Conv1 backward
    conv_backward(cnn->conv1, input, cnn->conv1_relu_grad, batch_size);
    
    free(fc2_grad);
}

// Update weights using SGD (all layers)
void update_weights(CNN* cnn, float learning_rate) {
    update_encoder_weights(cnn, learning_rate);
    update_classifier_weights(cnn, learning_rate);
}

// Update classifier weights (FC1, FC2)
void update_classifier_weights(CNN* cnn, float learning_rate) {
    // Update FC1
    int fc1_weight_size = FC1_OUTPUT_SIZE * FC1_INPUT_SIZE;
    for (int i = 0; i < fc1_weight_size; i++) {
        cnn->fc1->weights[i] -= learning_rate * cnn->fc1->weight_gradients[i];
    }
    for (int i = 0; i < FC1_OUTPUT_SIZE; i++) {
        cnn->fc1->bias[i] -= learning_rate * cnn->fc1->bias_gradients[i];
    }
    
    // Update FC2
    int fc2_weight_size = FC2_OUTPUT_SIZE * FC2_INPUT_SIZE;
    for (int i = 0; i < fc2_weight_size; i++) {
        cnn->fc2->weights[i] -= learning_rate * cnn->fc2->weight_gradients[i];
    }
    for (int i = 0; i < FC2_OUTPUT_SIZE; i++) {
        cnn->fc2->bias[i] -= learning_rate * cnn->fc2->bias_gradients[i];
    }
}

// Update encoder weights (Conv1, Conv2) with accumulated gradients
void update_encoder_weights(CNN* cnn, float learning_rate) {
    // Update Conv1
    int conv1_weight_size = CONV1_FILTERS * INPUT_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE;
    for (int i = 0; i < conv1_weight_size; i++) {
        cnn->conv1->weights[i] -= learning_rate * cnn->conv1->weight_gradients[i];
    }
    for (int i = 0; i < CONV1_FILTERS; i++) {
        cnn->conv1->bias[i] -= learning_rate * cnn->conv1->bias_gradients[i];
    }
    
    // Update Conv2
    int conv2_weight_size = CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE;
    for (int i = 0; i < conv2_weight_size; i++) {
        cnn->conv2->weights[i] -= learning_rate * cnn->conv2->weight_gradients[i];
    }
    for (int i = 0; i < CONV2_FILTERS; i++) {
        cnn->conv2->bias[i] -= learning_rate * cnn->conv2->bias_gradients[i];
    }
}

// Backpropagate reconstruction gradients to encoder (accumulates with classification gradients)
void backprop_reconstruction_to_encoder(CNN* cnn, float* input, float* pool2_gradient, int batch_size) {
    // Backward through Pool2 (accumulate gradients)
    maxpool_backward(cnn->pool2, pool2_gradient, batch_size);
    
    // Backward through ReLU for Conv2 (accumulate)
    int conv2_size = batch_size * CONV2_FILTERS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE;
    for (int i = 0; i < conv2_size; i++) {
        if (cnn->conv2_relu[i] > 0) {
            cnn->conv2_relu_grad[i] += cnn->pool2->input_gradients[i];
        }
    }
    
    // Backward through Conv2 (accumulate gradients)
    conv_backward(cnn->conv2, cnn->pool1->output, cnn->conv2_relu_grad, batch_size);
    
    // Backward through Pool1 (accumulate)
    maxpool_backward(cnn->pool1, cnn->conv2->input_gradients, batch_size);
    
    // Backward through ReLU for Conv1 (accumulate)
    int conv1_size = batch_size * CONV1_FILTERS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE;
    for (int i = 0; i < conv1_size; i++) {
        if (cnn->conv1_relu[i] > 0) {
            cnn->conv1_relu_grad[i] += cnn->pool1->input_gradients[i];
        }
    }
    
    // Backward through Conv1 (accumulate gradients)
    conv_backward(cnn->conv1, input, cnn->conv1_relu_grad, batch_size);
}

// Transpose convolution backward pass
void transpose_conv_backward(TransposeConvLayer* layer, float* input, float* output_gradient, int batch_size) {
    int input_size = layer->input_size;
    int output_size = layer->output_size;
    int kernel_size = layer->kernel_size;
    int padding = layer->padding;
    int stride = layer->stride;
    int input_channels = layer->input_channels;
    int output_channels = layer->output_channels;
    
    // Clear gradients
    memset(layer->weight_gradients, 0, output_channels * input_channels * kernel_size * kernel_size * sizeof(float));
    memset(layer->bias_gradients, 0, output_channels * sizeof(float));
    memset(layer->input_gradients, 0, batch_size * input_channels * input_size * input_size * sizeof(float));
    
    // For each image in batch
    for (int b = 0; b < batch_size; b++) {
        // For each input position
        for (int ih = 0; ih < input_size; ih++) {
            for (int iw = 0; iw < input_size; iw++) {
                // For each input channel
                for (int ic = 0; ic < input_channels; ic++) {
                    int input_idx = b * (input_channels * input_size * input_size) +
                                   ic * (input_size * input_size) +
                                   ih * input_size + iw;
                    float input_val = input[input_idx];
                    
                    // For each output channel
                    for (int oc = 0; oc < output_channels; oc++) {
                        // For each kernel position
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int oh = ih * stride + kh - padding;
                                int ow = iw * stride + kw - padding;
                                
                                // Check bounds
                                if (oh >= 0 && oh < output_size && ow >= 0 && ow < output_size) {
                                    int output_idx = b * (output_channels * output_size * output_size) +
                                                    oc * (output_size * output_size) +
                                                    oh * output_size + ow;
                                    int weight_idx = oc * (input_channels * kernel_size * kernel_size) +
                                                    ic * (kernel_size * kernel_size) +
                                                    kh * kernel_size + kw;
                                    
                                    float out_grad = output_gradient[output_idx];
                                    
                                    // Weight gradient
                                    layer->weight_gradients[weight_idx] += out_grad * input_val;
                                    
                                    // Input gradient
                                    layer->input_gradients[input_idx] += out_grad * layer->weights[weight_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Bias gradients
        for (int oc = 0; oc < output_channels; oc++) {
            for (int oh = 0; oh < output_size; oh++) {
                for (int ow = 0; ow < output_size; ow++) {
                    int output_idx = b * (output_channels * output_size * output_size) +
                                    oc * (output_size * output_size) +
                                    oh * output_size + ow;
                    layer->bias_gradients[oc] += output_gradient[output_idx];
                }
            }
        }
    }
}

// Upsample backward pass
void upsample_backward(UpsampleLayer* layer, float* output_gradient, int batch_size) {
    int input_size = layer->input_size;
    int output_size = layer->output_size;
    int channels = layer->channels;
    int scale = layer->scale_factor;
    
    // Clear input gradients
    memset(layer->input_gradients, 0, batch_size * channels * input_size * input_size * sizeof(float));
    
    // Sum gradients from all output positions that came from each input position
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < output_size; oh++) {
                for (int ow = 0; ow < output_size; ow++) {
                    int ih = oh / scale;
                    int iw = ow / scale;
                    int input_idx = b * (channels * input_size * input_size) +
                                   c * (input_size * input_size) +
                                   ih * input_size + iw;
                    int output_idx = b * (channels * output_size * output_size) +
                                    c * (output_size * output_size) +
                                    oh * output_size + ow;
                    layer->input_gradients[input_idx] += output_gradient[output_idx];
                }
            }
        }
    }
}

// Decoder backward pass
void decoder_backward(Decoder* decoder, float* input, float* output_gradient) {
    int batch_size = decoder->batch_size;
    
    // Backward through TransConv2
    transpose_conv_backward(decoder->tconv2, decoder->upsample2->output, output_gradient, batch_size);
    
    // Backward through Upsample2
    upsample_backward(decoder->upsample2, decoder->tconv2->input_gradients, batch_size);
    
    // Backward through ReLU
    relu_backward(decoder->tconv1->output, decoder->upsample2->input_gradients,
                  decoder->tconv1_relu_grad, batch_size * 64 * 16 * 16);
    
    // Backward through TransConv1
    transpose_conv_backward(decoder->tconv1, decoder->upsample1->output, decoder->tconv1_relu_grad, batch_size);
    
    // Backward through Upsample1
    upsample_backward(decoder->upsample1, decoder->tconv1->input_gradients, batch_size);
}

// Update decoder weights using gradients (SGD)
void update_decoder_weights(Decoder* decoder, float learning_rate) {
    // Update TransConv1
    int tconv1_weight_size = 64 * 128 * 3 * 3;
    for (int i = 0; i < tconv1_weight_size; i++) {
        decoder->tconv1->weights[i] -= learning_rate * decoder->tconv1->weight_gradients[i];
    }
    for (int i = 0; i < 64; i++) {
        decoder->tconv1->bias[i] -= learning_rate * decoder->tconv1->bias_gradients[i];
    }
    
    // Update TransConv2
    int tconv2_weight_size = 3 * 64 * 3 * 3;
    for (int i = 0; i < tconv2_weight_size; i++) {
        decoder->tconv2->weights[i] -= learning_rate * decoder->tconv2->weight_gradients[i];
    }
    for (int i = 0; i < 3; i++) {
        decoder->tconv2->bias[i] -= learning_rate * decoder->tconv2->bias_gradients[i];
    }
}
