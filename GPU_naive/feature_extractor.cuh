#ifndef FEATURE_EXTRACTOR_CUH
#define FEATURE_EXTRACTOR_CUH

#include "cnn.cuh"
#include "data_loader.h"

// Feature extractor using encoder part of CNN (Conv1->Pool1->Conv2->Pool2)
// Extracts features of size: CONV2_FILTERS * POOL2_OUTPUT_SIZE * POOL2_OUTPUT_SIZE (128*8*8 = 8192)
#define FEATURE_SIZE (CONV2_FILTERS * POOL2_OUTPUT_SIZE * POOL2_OUTPUT_SIZE)

// Save encoder weights to file
int save_encoder_weights(CNN* cnn, const char* filename);

// Load encoder weights from file
int load_encoder_weights(CNN* cnn, const char* filename);

// Extract features from dataset using encoder (forward pass only through Conv1->Pool1->Conv2->Pool2)
// Returns array of features: num_samples x FEATURE_SIZE
float* extract_features(CNN* cnn, CIFAR10_Dataset* dataset, int batch_size);

// Run encoder forward pass only (Conv1->Pool1->Conv2->Pool2)
void encoder_forward_pass(CNN* cnn, float* d_input);

// Save/load features to/from file
int save_features(const char* filename, float* features, int num_samples, int feature_size);
float* load_features(const char* filename, int* num_samples, int* feature_size);

#endif // FEATURE_EXTRACTOR_CUH
