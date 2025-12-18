#include "forward.h"
#include <math.h>
#include <string.h>
#include <float.h>

// Convolution forward pass
void conv_forward(ConvLayer* layer, float* input, int batch_size) {
    int input_size = layer->input_size;
    int output_size = layer->output_size;
    int kernel_size = layer->kernel_size;
    int padding = layer->padding;
    int stride = layer->stride;
    int input_channels = layer->input_channels;
    int output_channels = layer->output_channels;
    
    // Clear output
    memset(layer->output, 0, batch_size * output_channels * output_size * output_size * sizeof(float));
    
    // For each image in batch
    for (int b = 0; b < batch_size; b++) {
        // For each output channel
        for (int oc = 0; oc < output_channels; oc++) {
            // For each output position
            for (int oh = 0; oh < output_size; oh++) {
                for (int ow = 0; ow < output_size; ow++) {
                    float sum = layer->bias[oc];
                    
                    // For each input channel
                    for (int ic = 0; ic < input_channels; ic++) {
                        // For each kernel position
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;
                                
                                // Check bounds
                                if (ih >= 0 && ih < input_size && iw >= 0 && iw < input_size) {
                                    int input_idx = b * (input_channels * input_size * input_size) +
                                                   ic * (input_size * input_size) +
                                                   ih * input_size + iw;
                                    int weight_idx = oc * (input_channels * kernel_size * kernel_size) +
                                                    ic * (kernel_size * kernel_size) +
                                                    kh * kernel_size + kw;
                                    sum += input[input_idx] * layer->weights[weight_idx];
                                }
                            }
                        }
                    }
                    
                    int output_idx = b * (output_channels * output_size * output_size) +
                                    oc * (output_size * output_size) +
                                    oh * output_size + ow;
                    layer->output[output_idx] = sum;
                }
            }
        }
    }
}

// ReLU activation
void relu_forward(float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] > 0) ? input[i] : 0;
    }
}

// Max pooling forward pass
void maxpool_forward(MaxPoolLayer* layer, float* input, int batch_size) {
    int input_size = layer->input_size;
    int output_size = layer->output_size;
    int pool_size = layer->pool_size;
    int stride = layer->stride;
    int channels = layer->input_channels;
    
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < output_size; oh++) {
                for (int ow = 0; ow < output_size; ow++) {
                    float max_val = -FLT_MAX;
                    int max_idx = 0;
                    
                    // Find max in pool region
                    for (int ph = 0; ph < pool_size; ph++) {
                        for (int pw = 0; pw < pool_size; pw++) {
                            int ih = oh * stride + ph;
                            int iw = ow * stride + pw;
                            
                            int input_idx = b * (channels * input_size * input_size) +
                                          c * (input_size * input_size) +
                                          ih * input_size + iw;
                            
                            if (input[input_idx] > max_val) {
                                max_val = input[input_idx];
                                max_idx = input_idx;
                            }
                        }
                    }
                    
                    int output_idx = b * (channels * output_size * output_size) +
                                    c * (output_size * output_size) +
                                    oh * output_size + ow;
                    layer->output[output_idx] = max_val;
                    layer->max_indices[output_idx] = max_idx;
                }
            }
        }
    }
}

// Fully connected forward pass
void fc_forward(FCLayer* layer, float* input, int batch_size) {
    int input_size = layer->input_size;
    int output_size = layer->output_size;
    
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < output_size; o++) {
            float sum = layer->bias[o];
            for (int i = 0; i < input_size; i++) {
                sum += input[b * input_size + i] * layer->weights[o * input_size + i];
            }
            layer->output[b * output_size + o] = sum;
        }
    }
}

// Softmax activation
void softmax_forward(float* input, float* output, int batch_size, int num_classes) {
    for (int b = 0; b < batch_size; b++) {
        // Find max for numerical stability
        float max_val = input[b * num_classes];
        for (int i = 1; i < num_classes; i++) {
            if (input[b * num_classes + i] > max_val) {
                max_val = input[b * num_classes + i];
            }
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            output[b * num_classes + i] = expf(input[b * num_classes + i] - max_val);
            sum += output[b * num_classes + i];
        }
        
        // Normalize
        for (int i = 0; i < num_classes; i++) {
            output[b * num_classes + i] /= sum;
        }
    }
}

// Upsample forward pass (nearest neighbor)
void upsample_forward(UpsampleLayer* layer, float* input, int batch_size) {
    int input_size = layer->input_size;
    int output_size = layer->output_size;
    int channels = layer->channels;
    int scale = layer->scale_factor;
    
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
                    layer->output[output_idx] = input[input_idx];
                }
            }
        }
    }
}

// Decoder forward pass
void decoder_forward(Decoder* decoder, float* input) {
    int batch_size = decoder->batch_size;
    
    // Conv1 + ReLU: 128ch → 128ch, 8×8
    conv_forward(decoder->conv1, input, batch_size);
    relu_forward(decoder->conv1->output, decoder->conv1_relu,
                 batch_size * 128 * 8 * 8);
    
    // Upsample1: 8×8 → 16×16 (128 channels)
    upsample_forward(decoder->upsample1, decoder->conv1_relu, batch_size);
    
    // Conv2 + ReLU: 128ch → 256ch, 16×16
    conv_forward(decoder->conv2, decoder->upsample1->output, batch_size);
    relu_forward(decoder->conv2->output, decoder->conv2_relu,
                 batch_size * 256 * 16 * 16);
    
    // Upsample2: 16×16 → 32×32
    upsample_forward(decoder->upsample2, decoder->conv2_relu, batch_size);
    
    // Conv3: 64ch → 3ch, 32×32 (final reconstruction)
    conv_forward(decoder->conv3, decoder->upsample2->output, batch_size);
    
    // Copy to reconstructed output
    memcpy(decoder->reconstructed, decoder->conv3->output, 
           batch_size * 3 * 32 * 32 * sizeof(float));
}

// Complete forward pass
void forward_pass(CNN* cnn, float* input) {
    int batch_size = cnn->batch_size;
    
    // Conv1 + ReLU
    conv_forward(cnn->conv1, input, batch_size);
    relu_forward(cnn->conv1->output, cnn->conv1_relu,
                 batch_size * CONV1_FILTERS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE);
    
    // Pool1
    maxpool_forward(cnn->pool1, cnn->conv1_relu, batch_size);
    
    // Conv2 + ReLU
    conv_forward(cnn->conv2, cnn->pool1->output, batch_size);
    relu_forward(cnn->conv2->output, cnn->conv2_relu,
                 batch_size * CONV2_FILTERS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE);
    
    // Pool2
    maxpool_forward(cnn->pool2, cnn->conv2_relu, batch_size);
    
    // FC1 + ReLU
    fc_forward(cnn->fc1, cnn->pool2->output, batch_size);
    relu_forward(cnn->fc1->output, cnn->fc1_relu, batch_size * FC1_OUTPUT_SIZE);
    
    // FC2
    fc_forward(cnn->fc2, cnn->fc1_relu, batch_size);
    
    // Softmax
    softmax_forward(cnn->fc2->output, cnn->output, batch_size, FC2_OUTPUT_SIZE);
}
