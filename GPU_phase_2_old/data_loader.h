#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <stdint.h>

#define CIFAR10_IMAGE_SIZE 3072  // 32x32x3
#define CIFAR10_IMAGE_WIDTH 32
#define CIFAR10_IMAGE_HEIGHT 32
#define CIFAR10_IMAGE_CHANNELS 3
#define CIFAR10_NUM_CLASSES 10
#define CIFAR10_BATCH_SIZE 10000

// Structure to hold a single image
typedef struct {
    uint8_t label;
    uint8_t data[CIFAR10_IMAGE_SIZE];
} CIFAR10_Image;

// Structure to hold a batch of images
typedef struct {
    int num_images;
    CIFAR10_Image* images;
} CIFAR10_Batch;

// Structure to hold normalized floating-point data for training
typedef struct {
    int num_samples;
    float* images;    // num_samples x 3072, normalized to [0, 1]
    uint8_t* labels;  // num_samples labels
} CIFAR10_Dataset;

// Function declarations
CIFAR10_Batch* load_cifar10_batch(const char* filename);
void free_cifar10_batch(CIFAR10_Batch* batch);

CIFAR10_Dataset* create_dataset(int num_samples);
void free_dataset(CIFAR10_Dataset* dataset);

// Load and normalize training data
CIFAR10_Dataset* load_training_data(const char* data_dir);
// Load and normalize test data
CIFAR10_Dataset* load_test_data(const char* data_dir);

// Normalize image data from [0, 255] to [0, 1]
void normalize_images(CIFAR10_Dataset* dataset, CIFAR10_Batch* batch, int offset);

#endif // DATA_LOADER_H
