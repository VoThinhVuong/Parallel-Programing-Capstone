#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include "cnn.h"
#include "data_loader.h"



#define FEATURE_SIZE (CONV2_FILTERS * POOL2_OUTPUT_SIZE * POOL2_OUTPUT_SIZE)


int save_encoder_weights(CNN* cnn, const char* filename);


int load_encoder_weights(CNN* cnn, const char* filename);



float* extract_features(CNN* cnn, CIFAR10_Dataset* dataset, int batch_size);


void encoder_forward_pass(CNN* cnn, float* input, int batch_size);


int save_features(const char* filename, float* features, int num_samples, int feature_size);
float* load_features(const char* filename, int* num_samples, int* feature_size);

#endif 
