#include "cnn.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>


ConvLayer* create_conv_layer(int input_channels, int output_channels,
                             int kernel_size, int stride, int padding,
                             int input_size, int batch_size) {
    ConvLayer* layer = (ConvLayer*)malloc(sizeof(ConvLayer));
    if (!layer) return NULL;
    
    layer->input_channels = input_channels;
    layer->output_channels = output_channels;
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding = padding;
    layer->input_size = input_size;
    layer->output_size = (input_size + 2 * padding - kernel_size) / stride + 1;
    
    int weight_size = output_channels * input_channels * kernel_size * kernel_size;
    int output_total_size = batch_size * output_channels * layer->output_size * layer->output_size;
    
    layer->weights = (float*)malloc(weight_size * sizeof(float));
    layer->bias = (float*)malloc(output_channels * sizeof(float));
    layer->output = (float*)malloc(output_total_size * sizeof(float));
    layer->weight_gradients = (float*)malloc(weight_size * sizeof(float));
    layer->bias_gradients = (float*)malloc(output_channels * sizeof(float));
    layer->input_gradients = (float*)malloc(batch_size * input_channels * input_size * input_size * sizeof(float));
    
    if (!layer->weights || !layer->bias || !layer->output || 
        !layer->weight_gradients || !layer->bias_gradients || !layer->input_gradients) {
        free_conv_layer(layer);
        return NULL;
    }
    
    
    memset(layer->weights, 0, weight_size * sizeof(float));
    memset(layer->bias, 0, output_channels * sizeof(float));
    memset(layer->weight_gradients, 0, weight_size * sizeof(float));
    memset(layer->bias_gradients, 0, output_channels * sizeof(float));
    
    return layer;
}


MaxPoolLayer* create_maxpool_layer(int pool_size, int stride,
                                   int input_channels, int input_size,
                                   int batch_size) {
    MaxPoolLayer* layer = (MaxPoolLayer*)malloc(sizeof(MaxPoolLayer));
    if (!layer) return NULL;
    
    layer->pool_size = pool_size;
    layer->stride = stride;
    layer->input_channels = input_channels;
    layer->input_size = input_size;
    layer->output_size = (input_size - pool_size) / stride + 1;
    
    int output_total_size = batch_size * input_channels * layer->output_size * layer->output_size;
    int input_total_size = batch_size * input_channels * input_size * input_size;
    
    layer->output = (float*)malloc(output_total_size * sizeof(float));
    layer->max_indices = (int*)malloc(output_total_size * sizeof(int));
    layer->input_gradients = (float*)malloc(input_total_size * sizeof(float));
    
    if (!layer->output || !layer->max_indices || !layer->input_gradients) {
        free_maxpool_layer(layer);
        return NULL;
    }
    
    return layer;
}


FCLayer* create_fc_layer(int input_size, int output_size, int batch_size) {
    FCLayer* layer = (FCLayer*)malloc(sizeof(FCLayer));
    if (!layer) return NULL;
    
    layer->input_size = input_size;
    layer->output_size = output_size;
    
    layer->weights = (float*)malloc(output_size * input_size * sizeof(float));
    layer->bias = (float*)malloc(output_size * sizeof(float));
    layer->output = (float*)malloc(batch_size * output_size * sizeof(float));
    layer->weight_gradients = (float*)malloc(output_size * input_size * sizeof(float));
    layer->bias_gradients = (float*)malloc(output_size * sizeof(float));
    layer->input_gradients = (float*)malloc(batch_size * input_size * sizeof(float));
    
    if (!layer->weights || !layer->bias || !layer->output ||
        !layer->weight_gradients || !layer->bias_gradients || !layer->input_gradients) {
        free_fc_layer(layer);
        return NULL;
    }
    
    
    memset(layer->weights, 0, output_size * input_size * sizeof(float));
    memset(layer->bias, 0, output_size * sizeof(float));
    memset(layer->weight_gradients, 0, output_size * input_size * sizeof(float));
    memset(layer->bias_gradients, 0, output_size * sizeof(float));
    
    return layer;
}


void free_conv_layer(ConvLayer* layer) {
    if (!layer) return;
    if (layer->weights) free(layer->weights);
    if (layer->bias) free(layer->bias);
    if (layer->output) free(layer->output);
    if (layer->weight_gradients) free(layer->weight_gradients);
    if (layer->bias_gradients) free(layer->bias_gradients);
    if (layer->input_gradients) free(layer->input_gradients);
    free(layer);
}

void free_maxpool_layer(MaxPoolLayer* layer) {
    if (!layer) return;
    if (layer->output) free(layer->output);
    if (layer->max_indices) free(layer->max_indices);
    if (layer->input_gradients) free(layer->input_gradients);
    free(layer);
}

void free_fc_layer(FCLayer* layer) {
    if (!layer) return;
    if (layer->weights) free(layer->weights);
    if (layer->bias) free(layer->bias);
    if (layer->output) free(layer->output);
    if (layer->weight_gradients) free(layer->weight_gradients);
    if (layer->bias_gradients) free(layer->bias_gradients);
    if (layer->input_gradients) free(layer->input_gradients);
    free(layer);
}


CNN* create_cnn(int batch_size) {
    CNN* cnn = (CNN*)malloc(sizeof(CNN));
    if (!cnn) return NULL;
    
    cnn->batch_size = batch_size;
    
    
    cnn->conv1 = create_conv_layer(INPUT_CHANNELS, CONV1_FILTERS, CONV1_KERNEL_SIZE,
                                   CONV1_STRIDE, CONV1_PADDING, INPUT_WIDTH, batch_size);
    cnn->pool1 = create_maxpool_layer(POOL1_SIZE, POOL1_STRIDE, CONV1_FILTERS,
                                     CONV1_OUTPUT_SIZE, batch_size);
    cnn->conv2 = create_conv_layer(CONV1_FILTERS, CONV2_FILTERS, CONV2_KERNEL_SIZE,
                                   CONV2_STRIDE, CONV2_PADDING, POOL1_OUTPUT_SIZE, batch_size);
    cnn->pool2 = create_maxpool_layer(POOL2_SIZE, POOL2_STRIDE, CONV2_FILTERS,
                                     CONV2_OUTPUT_SIZE, batch_size);
    cnn->fc1 = create_fc_layer(FC1_INPUT_SIZE, FC1_OUTPUT_SIZE, batch_size);
    cnn->fc2 = create_fc_layer(FC2_INPUT_SIZE, FC2_OUTPUT_SIZE, batch_size);
    
    
    cnn->conv1_relu = (float*)malloc(batch_size * CONV1_FILTERS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE * sizeof(float));
    cnn->conv2_relu = (float*)malloc(batch_size * CONV2_FILTERS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE * sizeof(float));
    cnn->fc1_relu = (float*)malloc(batch_size * FC1_OUTPUT_SIZE * sizeof(float));
    cnn->output = (float*)malloc(batch_size * FC2_OUTPUT_SIZE * sizeof(float));
    
    
    cnn->conv1_relu_grad = (float*)malloc(batch_size * CONV1_FILTERS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE * sizeof(float));
    cnn->conv2_relu_grad = (float*)malloc(batch_size * CONV2_FILTERS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE * sizeof(float));
    cnn->fc1_relu_grad = (float*)malloc(batch_size * FC1_OUTPUT_SIZE * sizeof(float));
    
    
    cnn->decoder = NULL;
    
    
    if (!cnn->conv1 || !cnn->pool1 || !cnn->conv2 || !cnn->pool2 || 
        !cnn->fc1 || !cnn->fc2 || !cnn->conv1_relu || !cnn->conv2_relu || 
        !cnn->fc1_relu || !cnn->output || !cnn->conv1_relu_grad ||
        !cnn->conv2_relu_grad || !cnn->fc1_relu_grad) {
        free_cnn(cnn);
        return NULL;
    }
    
    return cnn;
}


void free_cnn(CNN* cnn) {
    if (!cnn) return;
    
    free_conv_layer(cnn->conv1);
    free_maxpool_layer(cnn->pool1);
    free_conv_layer(cnn->conv2);
    free_maxpool_layer(cnn->pool2);
    free_fc_layer(cnn->fc1);
    free_fc_layer(cnn->fc2);
    
    if (cnn->decoder) free_decoder(cnn->decoder);
    
    if (cnn->conv1_relu) free(cnn->conv1_relu);
    if (cnn->conv2_relu) free(cnn->conv2_relu);
    if (cnn->fc1_relu) free(cnn->fc1_relu);
    if (cnn->output) free(cnn->output);
    if (cnn->conv1_relu_grad) free(cnn->conv1_relu_grad);
    if (cnn->conv2_relu_grad) free(cnn->conv2_relu_grad);
    if (cnn->fc1_relu_grad) free(cnn->fc1_relu_grad);
    
    free(cnn);
}


TransposeConvLayer* create_transpose_conv_layer(int input_channels, int output_channels,
                                                int kernel_size, int stride, int padding,
                                                int input_size, int batch_size) {
    TransposeConvLayer* layer = (TransposeConvLayer*)malloc(sizeof(TransposeConvLayer));
    if (!layer) return NULL;
    
    layer->input_channels = input_channels;
    layer->output_channels = output_channels;
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding = padding;
    layer->input_size = input_size;
    layer->output_size = (input_size - 1) * stride - 2 * padding + kernel_size;
    
    int weight_size = output_channels * input_channels * kernel_size * kernel_size;
    int output_total_size = batch_size * output_channels * layer->output_size * layer->output_size;
    
    layer->weights = (float*)malloc(weight_size * sizeof(float));
    layer->bias = (float*)malloc(output_channels * sizeof(float));
    layer->output = (float*)malloc(output_total_size * sizeof(float));
    layer->weight_gradients = (float*)malloc(weight_size * sizeof(float));
    layer->bias_gradients = (float*)malloc(output_channels * sizeof(float));
    layer->input_gradients = (float*)malloc(batch_size * input_channels * input_size * input_size * sizeof(float));
    
    if (!layer->weights || !layer->bias || !layer->output || 
        !layer->weight_gradients || !layer->bias_gradients || !layer->input_gradients) {
        free_transpose_conv_layer(layer);
        return NULL;
    }
    
    memset(layer->weights, 0, weight_size * sizeof(float));
    memset(layer->bias, 0, output_channels * sizeof(float));
    memset(layer->weight_gradients, 0, weight_size * sizeof(float));
    memset(layer->bias_gradients, 0, output_channels * sizeof(float));
    
    return layer;
}


UpsampleLayer* create_upsample_layer(int channels, int input_size, 
                                     int scale_factor, int batch_size) {
    UpsampleLayer* layer = (UpsampleLayer*)malloc(sizeof(UpsampleLayer));
    if (!layer) return NULL;
    
    layer->channels = channels;
    layer->input_size = input_size;
    layer->scale_factor = scale_factor;
    layer->output_size = input_size * scale_factor;
    
    int output_total_size = batch_size * channels * layer->output_size * layer->output_size;
    int input_total_size = batch_size * channels * input_size * input_size;
    
    layer->output = (float*)malloc(output_total_size * sizeof(float));
    layer->input_gradients = (float*)malloc(input_total_size * sizeof(float));
    
    if (!layer->output || !layer->input_gradients) {
        free_upsample_layer(layer);
        return NULL;
    }
    
    return layer;
}

void free_transpose_conv_layer(TransposeConvLayer* layer) {
    if (!layer) return;
    if (layer->weights) free(layer->weights);
    if (layer->bias) free(layer->bias);
    if (layer->output) free(layer->output);
    if (layer->weight_gradients) free(layer->weight_gradients);
    if (layer->bias_gradients) free(layer->bias_gradients);
    if (layer->input_gradients) free(layer->input_gradients);
    free(layer);
}

void free_upsample_layer(UpsampleLayer* layer) {
    if (!layer) return;
    if (layer->output) free(layer->output);
    if (layer->input_gradients) free(layer->input_gradients);
    free(layer);
}


Decoder* create_decoder(int batch_size) {
    Decoder* decoder = (Decoder*)malloc(sizeof(Decoder));
    if (!decoder) return NULL;
    
    decoder->batch_size = batch_size;
    
    
    decoder->upsample1 = create_upsample_layer(128, 8, 2, batch_size);
    
    
    decoder->tconv1 = create_transpose_conv_layer(128, 64, 3, 1, 1, 16, batch_size);
    
    
    decoder->upsample2 = create_upsample_layer(64, 16, 2, batch_size);
    
    
    decoder->tconv2 = create_transpose_conv_layer(64, 3, 3, 1, 1, 32, batch_size);
    
    
    decoder->tconv1_relu = (float*)malloc(batch_size * 64 * 16 * 16 * sizeof(float));
    decoder->tconv1_relu_grad = (float*)malloc(batch_size * 64 * 16 * 16 * sizeof(float));
    decoder->reconstructed = (float*)malloc(batch_size * 3 * 32 * 32 * sizeof(float));
    
    if (!decoder->upsample1 || !decoder->tconv1 || !decoder->upsample2 || 
        !decoder->tconv2 || !decoder->tconv1_relu || !decoder->tconv1_relu_grad || 
        !decoder->reconstructed) {
        free_decoder(decoder);
        return NULL;
    }
    
    return decoder;
}

void free_decoder(Decoder* decoder) {
    if (!decoder) return;
    
    free_upsample_layer(decoder->upsample1);
    free_transpose_conv_layer(decoder->tconv1);
    free_upsample_layer(decoder->upsample2);
    free_transpose_conv_layer(decoder->tconv2);
    
    if (decoder->tconv1_relu) free(decoder->tconv1_relu);
    if (decoder->tconv1_relu_grad) free(decoder->tconv1_relu_grad);
    if (decoder->reconstructed) free(decoder->reconstructed);
    
    free(decoder);
}

void initialize_decoder_weights(Decoder* decoder) {
    
    int tconv1_weight_size = 64 * 128 * 3 * 3;
    float tconv1_std = sqrtf(2.0f / (128 * 3 * 3));
    for (int i = 0; i < tconv1_weight_size; i++) {
        decoder->tconv1->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * tconv1_std;
    }
    
    
    int tconv2_weight_size = 3 * 64 * 3 * 3;
    float tconv2_std = sqrtf(2.0f / (64 * 3 * 3));
    for (int i = 0; i < tconv2_weight_size; i++) {
        decoder->tconv2->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * tconv2_std;
    }
}


void initialize_weights(CNN* cnn) {
    srand(time(NULL));
    
    
    int conv1_weight_size = CONV1_FILTERS * INPUT_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE;
    float conv1_std = sqrtf(2.0f / (INPUT_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE));
    for (int i = 0; i < conv1_weight_size; i++) {
        cnn->conv1->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * conv1_std;
    }
    
    
    int conv2_weight_size = CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE;
    float conv2_std = sqrtf(2.0f / (CONV1_FILTERS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE));
    for (int i = 0; i < conv2_weight_size; i++) {
        cnn->conv2->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * conv2_std;
    }
    
    
    int fc1_weight_size = FC1_OUTPUT_SIZE * FC1_INPUT_SIZE;
    float fc1_std = sqrtf(2.0f / FC1_INPUT_SIZE);
    for (int i = 0; i < fc1_weight_size; i++) {
        cnn->fc1->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * fc1_std;
    }
    
    
    int fc2_weight_size = FC2_OUTPUT_SIZE * FC2_INPUT_SIZE;
    float fc2_std = sqrtf(2.0f / FC2_INPUT_SIZE);
    for (int i = 0; i < fc2_weight_size; i++) {
        cnn->fc2->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * fc2_std;
    }
    
    printf("Weights initialized\n");
}
