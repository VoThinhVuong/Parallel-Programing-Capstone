#include "feature_extractor.cuh"
#include "cnn.cuh"
#include "forward.cuh"
#include "data_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Save encoder weights to binary file
int save_encoder_weights(CNN* cnn, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
        return -1;
    }
    
    printf("Saving encoder weights to %s...\n", filename);
    
    // Conv1 weights and bias
    int conv1_weight_size = CONV1_FILTERS * INPUT_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE;
    float* h_conv1_weights = (float*)malloc(conv1_weight_size * sizeof(float));
    float* h_conv1_bias = (float*)malloc(CONV1_FILTERS * sizeof(float));
    
    CUDA_CHECK(cudaMemcpy(h_conv1_weights, cnn->conv1->d_weights, 
                         conv1_weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_conv1_bias, cnn->conv1->d_bias, 
                         CONV1_FILTERS * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(&conv1_weight_size, sizeof(int), 1, file);
    fwrite(h_conv1_weights, sizeof(float), conv1_weight_size, file);
    int conv1_filters = CONV1_FILTERS;
    fwrite(&conv1_filters, sizeof(int), 1, file);
    fwrite(h_conv1_bias, sizeof(float), CONV1_FILTERS, file);
    
    free(h_conv1_weights);
    free(h_conv1_bias);
    
    // Conv2 weights and bias
    int conv2_weight_size = CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE;
    float* h_conv2_weights = (float*)malloc(conv2_weight_size * sizeof(float));
    float* h_conv2_bias = (float*)malloc(CONV2_FILTERS * sizeof(float));
    
    CUDA_CHECK(cudaMemcpy(h_conv2_weights, cnn->conv2->d_weights, 
                         conv2_weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_conv2_bias, cnn->conv2->d_bias, 
                         CONV2_FILTERS * sizeof(float), cudaMemcpyDeviceToHost));
    
    fwrite(&conv2_weight_size, sizeof(int), 1, file);
    fwrite(h_conv2_weights, sizeof(float), conv2_weight_size, file);
    int conv2_filters = CONV2_FILTERS;
    fwrite(&conv2_filters, sizeof(int), 1, file);
    fwrite(h_conv2_bias, sizeof(float), CONV2_FILTERS, file);
    
    free(h_conv2_weights);
    free(h_conv2_bias);
    
    fclose(file);
    printf("Encoder weights saved successfully\n");
    return 0;
}

// Load encoder weights from binary file
int load_encoder_weights(CNN* cnn, const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s for reading\n", filename);
        return -1;
    }
    
    printf("Loading encoder weights from %s...\n", filename);
    
    // Conv1 weights and bias
    int conv1_weight_size;
    fread(&conv1_weight_size, sizeof(int), 1, file);
    
    float* h_conv1_weights = (float*)malloc(conv1_weight_size * sizeof(float));
    fread(h_conv1_weights, sizeof(float), conv1_weight_size, file);
    
    int conv1_filters;
    fread(&conv1_filters, sizeof(int), 1, file);
    float* h_conv1_bias = (float*)malloc(conv1_filters * sizeof(float));
    fread(h_conv1_bias, sizeof(float), conv1_filters, file);
    
    CUDA_CHECK(cudaMemcpy(cnn->conv1->d_weights, h_conv1_weights, 
                         conv1_weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cnn->conv1->d_bias, h_conv1_bias, 
                         conv1_filters * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_conv1_weights);
    free(h_conv1_bias);
    
    // Conv2 weights and bias
    int conv2_weight_size;
    fread(&conv2_weight_size, sizeof(int), 1, file);
    
    float* h_conv2_weights = (float*)malloc(conv2_weight_size * sizeof(float));
    fread(h_conv2_weights, sizeof(float), conv2_weight_size, file);
    
    int conv2_filters;
    fread(&conv2_filters, sizeof(int), 1, file);
    float* h_conv2_bias = (float*)malloc(conv2_filters * sizeof(float));
    fread(h_conv2_bias, sizeof(float), conv2_filters, file);
    
    CUDA_CHECK(cudaMemcpy(cnn->conv2->d_weights, h_conv2_weights, 
                         conv2_weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cnn->conv2->d_bias, h_conv2_bias, 
                         conv2_filters * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_conv2_weights);
    free(h_conv2_bias);
    
    fclose(file);
    printf("Encoder weights loaded successfully\n");
    return 0;
}

// Run encoder forward pass only (Conv1->Pool1->Conv2->Pool2, no FC layers)
void encoder_forward_pass(CNN* cnn, float* d_input) {
    int batch_size = cnn->batch_size;
    
    // Conv1 + ReLU (in-place)
    conv_forward(cnn->conv1, d_input, batch_size);
    relu_forward(cnn->conv1->d_output,
                 batch_size * CONV1_FILTERS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE);
    
    // Pool1 (takes ReLU output from conv1->d_output)
    maxpool_forward(cnn->pool1, cnn->conv1->d_output, batch_size);
    
    // Conv2 + ReLU (in-place)
    conv_forward(cnn->conv2, cnn->pool1->d_output, batch_size);
    relu_forward(cnn->conv2->d_output,
                 batch_size * CONV2_FILTERS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE);
    
    // Pool2 (takes ReLU output from conv2->d_output, output is the feature representation)
    maxpool_forward(cnn->pool2, cnn->conv2->d_output, batch_size);
    
    // Features are now in cnn->pool2->d_output with size: batch_size x FEATURE_SIZE
}

// Extract features from entire dataset
float* extract_features(CNN* cnn, CIFAR10_Dataset* dataset, int batch_size) {
    int num_samples = dataset->num_samples;
    int num_batches = (num_samples + batch_size - 1) / batch_size;
    
    printf("Extracting features from %d samples...\n", num_samples);
    
    // Allocate host memory for all features
    float* h_features = (float*)malloc(num_samples * FEATURE_SIZE * sizeof(float));
    if (!h_features) {
        fprintf(stderr, "Error: Failed to allocate memory for features\n");
        return NULL;
    }
    
    // Allocate device memory for batch
    float* d_batch_images;
    CUDA_CHECK(cudaMalloc(&d_batch_images, batch_size * CIFAR10_IMAGE_SIZE * sizeof(float)));
    
    // Process each batch
    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        int offset = batch_idx * batch_size;
        int current_batch_size = (offset + batch_size <= num_samples) ? batch_size : (num_samples - offset);
        
        // Copy batch to device
        CUDA_CHECK(cudaMemcpy(d_batch_images, 
                             &dataset->images[offset * CIFAR10_IMAGE_SIZE],
                             current_batch_size * CIFAR10_IMAGE_SIZE * sizeof(float),
                             cudaMemcpyHostToDevice));
        
        // Run encoder forward pass
        encoder_forward_pass(cnn, d_batch_images);
        
        // Copy features back to host
        CUDA_CHECK(cudaMemcpy(&h_features[offset * FEATURE_SIZE],
                             cnn->pool2->d_output,
                             current_batch_size * FEATURE_SIZE * sizeof(float),
                             cudaMemcpyDeviceToHost));
        
        // Progress indicator
        if ((batch_idx + 1) % 10 == 0 || batch_idx == num_batches - 1) {
            printf("  Processed %d/%d batches (%.1f%%)\r", 
                   batch_idx + 1, num_batches, 
                   100.0f * (batch_idx + 1) / num_batches);
            fflush(stdout);
        }
    }
    
    printf("\nFeature extraction complete: %d samples x %d features\n", 
           num_samples, FEATURE_SIZE);
    
    CUDA_CHECK(cudaFree(d_batch_images));
    
    return h_features;
}

// Save features to binary file
int save_features(const char* filename, float* features, int num_samples, int feature_size) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
        return -1;
    }
    
    printf("Saving features to %s...\n", filename);
    
    fwrite(&num_samples, sizeof(int), 1, file);
    fwrite(&feature_size, sizeof(int), 1, file);
    fwrite(features, sizeof(float), num_samples * feature_size, file);
    
    fclose(file);
    printf("Features saved: %d samples x %d features\n", num_samples, feature_size);
    return 0;
}

// Load features from binary file
float* load_features(const char* filename, int* num_samples, int* feature_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s for reading\n", filename);
        return NULL;
    }
    
    printf("Loading features from %s...\n", filename);
    
    fread(num_samples, sizeof(int), 1, file);
    fread(feature_size, sizeof(int), 1, file);
    
    float* features = (float*)malloc((*num_samples) * (*feature_size) * sizeof(float));
    if (!features) {
        fprintf(stderr, "Error: Failed to allocate memory for features\n");
        fclose(file);
        return NULL;
    }
    
    fread(features, sizeof(float), (*num_samples) * (*feature_size), file);
    
    fclose(file);
    printf("Features loaded: %d samples x %d features\n", *num_samples, *feature_size);
    return features;
}
