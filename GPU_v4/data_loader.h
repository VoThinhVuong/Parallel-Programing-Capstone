#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <stdint.h>

#define CIFAR10_IMAGE_SIZE 3072  
#define CIFAR10_IMAGE_WIDTH 32
#define CIFAR10_IMAGE_HEIGHT 32
#define CIFAR10_IMAGE_CHANNELS 3
#define CIFAR10_NUM_CLASSES 10
#define CIFAR10_BATCH_SIZE 10000


typedef struct {
    uint8_t label;
    uint8_t data[CIFAR10_IMAGE_SIZE];
} CIFAR10_Image;


typedef struct {
    int num_images;
    CIFAR10_Image* images;
} CIFAR10_Batch;


typedef struct {
    int num_samples;
    float* images;    
    uint8_t* labels;  
} CIFAR10_Dataset;


CIFAR10_Batch* load_cifar10_batch(const char* filename);
void free_cifar10_batch(CIFAR10_Batch* batch);

CIFAR10_Dataset* create_dataset(int num_samples);
void free_dataset(CIFAR10_Dataset* dataset);


CIFAR10_Dataset* load_training_data(const char* data_dir, int num_batches);

CIFAR10_Dataset* load_test_data(const char* data_dir);


void normalize_images(CIFAR10_Dataset* dataset, CIFAR10_Batch* batch, int offset);

#endif 
