#include "forward.h"
#include <math.h>
#include <string.h>
#include <float.h>


void conv_forward(ConvLayer* layer, float* input, int batch_size) {
    int input_size = layer->input_size;
    int output_size = layer->output_size;
    int kernel_size = layer->kernel_size;
    int padding = layer->padding;
    int stride = layer->stride;
    int input_channels = layer->input_channels;
    int output_channels = layer->output_channels;
    
    
    memset(layer->output, 0, batch_size * output_channels * output_size * output_size * sizeof(float));
    
    
    for (int b = 0; b < batch_size; b++) {
        
        for (int oc = 0; oc < output_channels; oc++) {
            
            for (int oh = 0; oh < output_size; oh++) {
                for (int ow = 0; ow < output_size; ow++) {
                    float sum = layer->bias[oc];
                    
                    
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


void relu_forward(float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] > 0) ? input[i] : 0;
    }
}


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


void softmax_forward(float* input, float* output, int batch_size, int num_classes) {
    for (int b = 0; b < batch_size; b++) {
        
        float max_val = input[b * num_classes];
        for (int i = 1; i < num_classes; i++) {
            if (input[b * num_classes + i] > max_val) {
                max_val = input[b * num_classes + i];
            }
        }
        
        
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            output[b * num_classes + i] = expf(input[b * num_classes + i] - max_val);
            sum += output[b * num_classes + i];
        }
        
        
        for (int i = 0; i < num_classes; i++) {
            output[b * num_classes + i] /= sum;
        }
    }
}


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


void transpose_conv_forward(TransposeConvLayer* layer, float* input, int batch_size) {
    int input_size = layer->input_size;
    int output_size = layer->output_size;
    int kernel_size = layer->kernel_size;
    int padding = layer->padding;
    int stride = layer->stride;
    int input_channels = layer->input_channels;
    int output_channels = layer->output_channels;
    
    
    memset(layer->output, 0, batch_size * output_channels * output_size * output_size * sizeof(float));
    
    
    for (int b = 0; b < batch_size; b++) {
        
        for (int ih = 0; ih < input_size; ih++) {
            for (int iw = 0; iw < input_size; iw++) {
                
                for (int ic = 0; ic < input_channels; ic++) {
                    int input_idx = b * (input_channels * input_size * input_size) +
                                   ic * (input_size * input_size) +
                                   ih * input_size + iw;
                    float input_val = input[input_idx];
                    
                    
                    for (int oc = 0; oc < output_channels; oc++) {
                        
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int oh = ih * stride + kh - padding;
                                int ow = iw * stride + kw - padding;
                                
                                
                                if (oh >= 0 && oh < output_size && ow >= 0 && ow < output_size) {
                                    int output_idx = b * (output_channels * output_size * output_size) +
                                                    oc * (output_size * output_size) +
                                                    oh * output_size + ow;
                                    int weight_idx = oc * (input_channels * kernel_size * kernel_size) +
                                                    ic * (kernel_size * kernel_size) +
                                                    kh * kernel_size + kw;
                                    layer->output[output_idx] += input_val * layer->weights[weight_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
        
        
        for (int oc = 0; oc < output_channels; oc++) {
            for (int oh = 0; oh < output_size; oh++) {
                for (int ow = 0; ow < output_size; ow++) {
                    int output_idx = b * (output_channels * output_size * output_size) +
                                    oc * (output_size * output_size) +
                                    oh * output_size + ow;
                    layer->output[output_idx] += layer->bias[oc];
                }
            }
        }
    }
}


void decoder_forward(Decoder* decoder, float* input) {
    int batch_size = decoder->batch_size;
    
    
    upsample_forward(decoder->upsample1, input, batch_size);
    
    
    transpose_conv_forward(decoder->tconv1, decoder->upsample1->output, batch_size);
    relu_forward(decoder->tconv1->output, decoder->tconv1_relu,
                 batch_size * 64 * 16 * 16);
    
    
    upsample_forward(decoder->upsample2, decoder->tconv1_relu, batch_size);
    
    
    transpose_conv_forward(decoder->tconv2, decoder->upsample2->output, batch_size);
    
    
    memcpy(decoder->reconstructed, decoder->tconv2->output, 
           batch_size * 3 * 32 * 32 * sizeof(float));
}


void forward_pass(CNN* cnn, float* input) {
    int batch_size = cnn->batch_size;
    
    
    conv_forward(cnn->conv1, input, batch_size);
    relu_forward(cnn->conv1->output, cnn->conv1_relu,
                 batch_size * CONV1_FILTERS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE);
    
    
    maxpool_forward(cnn->pool1, cnn->conv1_relu, batch_size);
    
    
    conv_forward(cnn->conv2, cnn->pool1->output, batch_size);
    relu_forward(cnn->conv2->output, cnn->conv2_relu,
                 batch_size * CONV2_FILTERS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE);
    
    
    maxpool_forward(cnn->pool2, cnn->conv2_relu, batch_size);
    
    
    fc_forward(cnn->fc1, cnn->pool2->output, batch_size);
    relu_forward(cnn->fc1->output, cnn->fc1_relu, batch_size * FC1_OUTPUT_SIZE);
    
    
    fc_forward(cnn->fc2, cnn->fc1_relu, batch_size);
    
    
    softmax_forward(cnn->fc2->output, cnn->output, batch_size, FC2_OUTPUT_SIZE);
}
