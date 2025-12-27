#include "cnn.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
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
    int input_grad_size = batch_size * input_channels * input_size * input_size;
    
    
    CUDA_CHECK(cudaMalloc(&layer->d_weights, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer->d_bias, output_channels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer->d_output, output_total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer->d_weight_gradients, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer->d_bias_gradients, output_channels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer->d_input_gradients, input_grad_size * sizeof(float)));
    
    
    CUDA_CHECK(cudaMemset(layer->d_weights, 0, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(layer->d_bias, 0, output_channels * sizeof(float)));
    CUDA_CHECK(cudaMemset(layer->d_weight_gradients, 0, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(layer->d_bias_gradients, 0, output_channels * sizeof(float)));
    
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
    
    
    CUDA_CHECK(cudaMalloc(&layer->d_output, output_total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer->d_max_indices, output_total_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&layer->d_input_gradients, input_total_size * sizeof(float)));
    
    return layer;
}


FCLayer* create_fc_layer(int input_size, int output_size, int batch_size) {
    FCLayer* layer = (FCLayer*)malloc(sizeof(FCLayer));
    if (!layer) return NULL;
    
    layer->input_size = input_size;
    layer->output_size = output_size;
    
    
    CUDA_CHECK(cudaMalloc(&layer->d_weights, output_size * input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer->d_bias, output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer->d_output, batch_size * output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer->d_weight_gradients, output_size * input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer->d_bias_gradients, output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer->d_input_gradients, batch_size * input_size * sizeof(float)));
    
    
    CUDA_CHECK(cudaMemset(layer->d_weights, 0, output_size * input_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(layer->d_bias, 0, output_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(layer->d_weight_gradients, 0, output_size * input_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(layer->d_bias_gradients, 0, output_size * sizeof(float)));
    
    return layer;
}


void free_conv_layer(ConvLayer* layer) {
    if (!layer) return;
    if (layer->d_weights) cudaFree(layer->d_weights);
    if (layer->d_bias) cudaFree(layer->d_bias);
    if (layer->d_output) cudaFree(layer->d_output);
    if (layer->d_weight_gradients) cudaFree(layer->d_weight_gradients);
    if (layer->d_bias_gradients) cudaFree(layer->d_bias_gradients);
    if (layer->d_input_gradients) cudaFree(layer->d_input_gradients);
    free(layer);
}

void free_maxpool_layer(MaxPoolLayer* layer) {
    if (!layer) return;
    if (layer->d_output) cudaFree(layer->d_output);
    if (layer->d_max_indices) cudaFree(layer->d_max_indices);
    if (layer->d_input_gradients) cudaFree(layer->d_input_gradients);
    free(layer);
}

void free_fc_layer(FCLayer* layer) {
    if (!layer) return;
    if (layer->d_weights) cudaFree(layer->d_weights);
    if (layer->d_bias) cudaFree(layer->d_bias);
    if (layer->d_output) cudaFree(layer->d_output);
    if (layer->d_weight_gradients) cudaFree(layer->d_weight_gradients);
    if (layer->d_bias_gradients) cudaFree(layer->d_bias_gradients);
    if (layer->d_input_gradients) cudaFree(layer->d_input_gradients);
    free(layer);
}


CNN* create_cnn(int batch_size) {
    CNN* cnn = (CNN*)malloc(sizeof(CNN));
    if (!cnn) return NULL;
    
    cnn->batch_size = batch_size;
    cnn->decoder = NULL;
    cnn->d_fc2_grad = NULL;
    
    
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
    
    
    CUDA_CHECK(cudaMalloc(&cnn->d_conv1_relu, batch_size * CONV1_FILTERS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cnn->d_conv2_relu, batch_size * CONV2_FILTERS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cnn->d_fc1_relu, batch_size * FC1_OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cnn->d_output, batch_size * FC2_OUTPUT_SIZE * sizeof(float)));
    
    
    CUDA_CHECK(cudaMalloc(&cnn->d_conv1_relu_grad, batch_size * CONV1_FILTERS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cnn->d_conv2_relu_grad, batch_size * CONV2_FILTERS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cnn->d_fc1_relu_grad, batch_size * FC1_OUTPUT_SIZE * sizeof(float)));

    
    CUDA_CHECK(cudaMalloc(&cnn->d_fc2_grad, batch_size * FC2_OUTPUT_SIZE * sizeof(float)));
    
    
    if (!cnn->conv1 || !cnn->pool1 || !cnn->conv2 || !cnn->pool2 || 
        !cnn->fc1 || !cnn->fc2) {
        free_cnn(cnn);
        return NULL;
    }
    
    return cnn;
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
    layer->output_size = (input_size - 1) * stride + kernel_size - 2 * padding;

    int weight_size = output_channels * input_channels * kernel_size * kernel_size;
    int output_total_size = batch_size * output_channels * layer->output_size * layer->output_size;
    int input_grad_size = batch_size * input_channels * input_size * input_size;

    CUDA_CHECK(cudaMalloc(&layer->d_weights, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer->d_bias, output_channels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer->d_output, output_total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer->d_weight_gradients, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer->d_bias_gradients, output_channels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer->d_input_gradients, input_grad_size * sizeof(float)));

    CUDA_CHECK(cudaMemset(layer->d_weights, 0, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(layer->d_bias, 0, output_channels * sizeof(float)));
    CUDA_CHECK(cudaMemset(layer->d_weight_gradients, 0, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(layer->d_bias_gradients, 0, output_channels * sizeof(float)));

    return layer;
}


UpsampleLayer* create_upsample_layer(int channels, int input_size, int scale_factor, int batch_size) {
    UpsampleLayer* layer = (UpsampleLayer*)malloc(sizeof(UpsampleLayer));
    if (!layer) return NULL;

    layer->channels = channels;
    layer->input_size = input_size;
    layer->scale_factor = scale_factor;
    layer->output_size = input_size * scale_factor;

    int output_total_size = batch_size * channels * layer->output_size * layer->output_size;
    int input_grad_size = batch_size * channels * input_size * input_size;

    CUDA_CHECK(cudaMalloc(&layer->d_output, output_total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer->d_input_gradients, input_grad_size * sizeof(float)));

    return layer;
}


Decoder* create_decoder(int batch_size) {
    Decoder* decoder = (Decoder*)malloc(sizeof(Decoder));
    if (!decoder) return NULL;

    decoder->batch_size = batch_size;

    decoder->upsample1 = create_upsample_layer(128, 8, 2, batch_size);
    decoder->tconv1 = create_transpose_conv_layer(128, 64, 3, 1, 1, 16, batch_size);
    decoder->upsample2 = create_upsample_layer(64, 16, 2, batch_size);
    decoder->tconv2 = create_transpose_conv_layer(64, 3, 3, 1, 1, 32, batch_size);

    CUDA_CHECK(cudaMalloc(&decoder->d_tconv1_relu, batch_size * 64 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&decoder->d_tconv1_relu_grad, batch_size * 64 * 16 * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&decoder->d_reconstructed, batch_size * 3 * 32 * 32 * sizeof(float)));

    if (!decoder->upsample1 || !decoder->tconv1 || !decoder->upsample2 || !decoder->tconv2) {
        free_decoder(decoder);
        return NULL;
    }

    return decoder;
}

void free_transpose_conv_layer(TransposeConvLayer* layer) {
    if (!layer) return;
    if (layer->d_weights) cudaFree(layer->d_weights);
    if (layer->d_bias) cudaFree(layer->d_bias);
    if (layer->d_output) cudaFree(layer->d_output);
    if (layer->d_weight_gradients) cudaFree(layer->d_weight_gradients);
    if (layer->d_bias_gradients) cudaFree(layer->d_bias_gradients);
    if (layer->d_input_gradients) cudaFree(layer->d_input_gradients);
    free(layer);
}

void free_upsample_layer(UpsampleLayer* layer) {
    if (!layer) return;
    if (layer->d_output) cudaFree(layer->d_output);
    if (layer->d_input_gradients) cudaFree(layer->d_input_gradients);
    free(layer);
}

void free_decoder(Decoder* decoder) {
    if (!decoder) return;
    free_upsample_layer(decoder->upsample1);
    free_transpose_conv_layer(decoder->tconv1);
    free_upsample_layer(decoder->upsample2);
    free_transpose_conv_layer(decoder->tconv2);
    if (decoder->d_tconv1_relu) cudaFree(decoder->d_tconv1_relu);
    if (decoder->d_tconv1_relu_grad) cudaFree(decoder->d_tconv1_relu_grad);
    if (decoder->d_reconstructed) cudaFree(decoder->d_reconstructed);
    free(decoder);
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
    
    if (cnn->d_conv1_relu) cudaFree(cnn->d_conv1_relu);
    if (cnn->d_conv2_relu) cudaFree(cnn->d_conv2_relu);
    if (cnn->d_fc1_relu) cudaFree(cnn->d_fc1_relu);
    if (cnn->d_output) cudaFree(cnn->d_output);
    if (cnn->d_fc2_grad) cudaFree(cnn->d_fc2_grad);
    if (cnn->d_conv1_relu_grad) cudaFree(cnn->d_conv1_relu_grad);
    if (cnn->d_conv2_relu_grad) cudaFree(cnn->d_conv2_relu_grad);
    if (cnn->d_fc1_relu_grad) cudaFree(cnn->d_fc1_relu_grad);
    
    free(cnn);
}


void initialize_weights(CNN* cnn) {
    srand(time(NULL));
    
    
    int conv1_weight_size = CONV1_FILTERS * INPUT_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE;
    float* h_conv1_weights = (float*)malloc(conv1_weight_size * sizeof(float));
    float conv1_std = sqrtf(2.0f / (INPUT_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE));
    for (int i = 0; i < conv1_weight_size; i++) {
        h_conv1_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * conv1_std;
    }
    CUDA_CHECK(cudaMemcpy(cnn->conv1->d_weights, h_conv1_weights, conv1_weight_size * sizeof(float), cudaMemcpyHostToDevice));
    free(h_conv1_weights);
    
    
    int conv2_weight_size = CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE;
    float* h_conv2_weights = (float*)malloc(conv2_weight_size * sizeof(float));
    float conv2_std = sqrtf(2.0f / (CONV1_FILTERS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE));
    for (int i = 0; i < conv2_weight_size; i++) {
        h_conv2_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * conv2_std;
    }
    CUDA_CHECK(cudaMemcpy(cnn->conv2->d_weights, h_conv2_weights, conv2_weight_size * sizeof(float), cudaMemcpyHostToDevice));
    free(h_conv2_weights);
    
    
    int fc1_weight_size = FC1_OUTPUT_SIZE * FC1_INPUT_SIZE;
    float* h_fc1_weights = (float*)malloc(fc1_weight_size * sizeof(float));
    float fc1_std = sqrtf(2.0f / FC1_INPUT_SIZE);
    for (int i = 0; i < fc1_weight_size; i++) {
        h_fc1_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * fc1_std;
    }
    CUDA_CHECK(cudaMemcpy(cnn->fc1->d_weights, h_fc1_weights, fc1_weight_size * sizeof(float), cudaMemcpyHostToDevice));
    free(h_fc1_weights);
    
    
    int fc2_weight_size = FC2_OUTPUT_SIZE * FC2_INPUT_SIZE;
    float* h_fc2_weights = (float*)malloc(fc2_weight_size * sizeof(float));
    float fc2_std = sqrtf(2.0f / FC2_INPUT_SIZE);
    for (int i = 0; i < fc2_weight_size; i++) {
        h_fc2_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * fc2_std;
    }
    CUDA_CHECK(cudaMemcpy(cnn->fc2->d_weights, h_fc2_weights, fc2_weight_size * sizeof(float), cudaMemcpyHostToDevice));
    free(h_fc2_weights);
    
    printf("Weights initialized and copied to GPU\n");
}

void initialize_decoder_weights(Decoder* decoder) {
    
    int tconv1_weight_size = 64 * 128 * 3 * 3;
    float* h_tconv1_weights = (float*)malloc(tconv1_weight_size * sizeof(float));
    float tconv1_std = sqrtf(2.0f / (128 * 3 * 3));
    for (int i = 0; i < tconv1_weight_size; i++) {
        h_tconv1_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * tconv1_std;
    }
    CUDA_CHECK(cudaMemcpy(decoder->tconv1->d_weights, h_tconv1_weights,
                          tconv1_weight_size * sizeof(float), cudaMemcpyHostToDevice));
    free(h_tconv1_weights);

    
    int tconv2_weight_size = 3 * 64 * 3 * 3;
    float* h_tconv2_weights = (float*)malloc(tconv2_weight_size * sizeof(float));
    float tconv2_std = sqrtf(2.0f / (64 * 3 * 3));
    for (int i = 0; i < tconv2_weight_size; i++) {
        h_tconv2_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * tconv2_std;
    }
    CUDA_CHECK(cudaMemcpy(decoder->tconv2->d_weights, h_tconv2_weights,
                          tconv2_weight_size * sizeof(float), cudaMemcpyHostToDevice));
    free(h_tconv2_weights);

    
    CUDA_CHECK(cudaMemset(decoder->tconv1->d_bias, 0, 64 * sizeof(float)));
    CUDA_CHECK(cudaMemset(decoder->tconv2->d_bias, 0, 3 * sizeof(float)));

    printf("Decoder weights initialized and copied to GPU\n");
}
