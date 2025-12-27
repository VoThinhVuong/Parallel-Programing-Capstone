#include "feature_extractor.h"
#include "forward.h"
#include "data_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int save_encoder_weights(CNN* cnn, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
        return -1;
    }
    
    printf("Saving encoder weights to %s...\n", filename);
    
    
    int conv1_weight_size = CONV1_FILTERS * INPUT_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE;
    fwrite(&conv1_weight_size, sizeof(int), 1, file);
    fwrite(cnn->conv1->weights, sizeof(float), conv1_weight_size, file);
    int conv1_filters = CONV1_FILTERS;
    fwrite(&conv1_filters, sizeof(int), 1, file);
    fwrite(cnn->conv1->bias, sizeof(float), CONV1_FILTERS, file);
    
    
    int conv2_weight_size = CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE;
    fwrite(&conv2_weight_size, sizeof(int), 1, file);
    fwrite(cnn->conv2->weights, sizeof(float), conv2_weight_size, file);
    int conv2_filters = CONV2_FILTERS;
    fwrite(&conv2_filters, sizeof(int), 1, file);
    fwrite(cnn->conv2->bias, sizeof(float), CONV2_FILTERS, file);
    
    fclose(file);
    printf("Encoder weights saved successfully\n");
    return 0;
}


int load_encoder_weights(CNN* cnn, const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s for reading\n", filename);
        return -1;
    }
    
    printf("Loading encoder weights from %s...\n", filename);
    
    
    int conv1_weight_size;
    fread(&conv1_weight_size, sizeof(int), 1, file);
    fread(cnn->conv1->weights, sizeof(float), conv1_weight_size, file);
    
    int conv1_filters;
    fread(&conv1_filters, sizeof(int), 1, file);
    fread(cnn->conv1->bias, sizeof(float), conv1_filters, file);
    
    
    int conv2_weight_size;
    fread(&conv2_weight_size, sizeof(int), 1, file);
    fread(cnn->conv2->weights, sizeof(float), conv2_weight_size, file);
    
    int conv2_filters;
    fread(&conv2_filters, sizeof(int), 1, file);
    fread(cnn->conv2->bias, sizeof(float), conv2_filters, file);
    
    fclose(file);
    printf("Encoder weights loaded successfully\n");
    return 0;
}


void encoder_forward_pass(CNN* cnn, float* input, int batch_size) {
    
    conv_forward(cnn->conv1, input, batch_size);
    relu_forward(cnn->conv1->output, cnn->conv1_relu,
                 batch_size * CONV1_FILTERS * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE);
    
    
    maxpool_forward(cnn->pool1, cnn->conv1_relu, batch_size);
    
    
    conv_forward(cnn->conv2, cnn->pool1->output, batch_size);
    relu_forward(cnn->conv2->output, cnn->conv2_relu,
                 batch_size * CONV2_FILTERS * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE);
    
    
    maxpool_forward(cnn->pool2, cnn->conv2_relu, batch_size);
    
    
}


float* extract_features(CNN* cnn, CIFAR10_Dataset* dataset, int batch_size) {
    int num_samples = dataset->num_samples;
    int num_batches = (num_samples + batch_size - 1) / batch_size;
    
    printf("Extracting features from %d samples...\n", num_samples);
    
    
    float* features = (float*)malloc(num_samples * FEATURE_SIZE * sizeof(float));
    if (!features) {
        fprintf(stderr, "Error: Failed to allocate memory for features\n");
        return NULL;
    }
    
    
    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        int offset = batch_idx * batch_size;
        int current_batch_size = (offset + batch_size <= num_samples) ? batch_size : (num_samples - offset);
        
        
        float* batch_images = &dataset->images[offset * CIFAR10_IMAGE_SIZE];
        
        
        encoder_forward_pass(cnn, batch_images, current_batch_size);
        
        
        for (int i = 0; i < current_batch_size; i++) {
            memcpy(&features[(offset + i) * FEATURE_SIZE],
                   &cnn->pool2->output[i * FEATURE_SIZE],
                   FEATURE_SIZE * sizeof(float));
        }
        
        
        if ((batch_idx + 1) % 10 == 0 || batch_idx == num_batches - 1) {
            printf("  Processed %d/%d batches (%.1f%%)\r", 
                   batch_idx + 1, num_batches, 
                   100.0f * (batch_idx + 1) / num_batches);
            fflush(stdout);
        }
    }
    
    printf("\nFeature extraction complete: %d samples x %d features\n", 
           num_samples, FEATURE_SIZE);
    
    return features;
}


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
