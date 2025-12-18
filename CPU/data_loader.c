#include "data_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Load a single CIFAR-10 batch file
CIFAR10_Batch* load_cifar10_batch(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return NULL;
    }

    CIFAR10_Batch* batch = (CIFAR10_Batch*)malloc(sizeof(CIFAR10_Batch));
    if (!batch) {
        fprintf(stderr, "Error: Memory allocation failed for batch\n");
        fclose(file);
        return NULL;
    }

    batch->num_images = CIFAR10_BATCH_SIZE;
    batch->images = (CIFAR10_Image*)malloc(CIFAR10_BATCH_SIZE * sizeof(CIFAR10_Image));
    if (!batch->images) {
        fprintf(stderr, "Error: Memory allocation failed for images\n");
        free(batch);
        fclose(file);
        return NULL;
    }

    // Read each image from the binary file
    for (int i = 0; i < CIFAR10_BATCH_SIZE; i++) {
        // Read label (1 byte)
        if (fread(&batch->images[i].label, 1, 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label for image %d\n", i);
            free(batch->images);
            free(batch);
            fclose(file);
            return NULL;
        }

        // Read image data (3072 bytes)
        if (fread(batch->images[i].data, 1, CIFAR10_IMAGE_SIZE, file) != CIFAR10_IMAGE_SIZE) {
            fprintf(stderr, "Error: Failed to read data for image %d\n", i);
            free(batch->images);
            free(batch);
            fclose(file);
            return NULL;
        }
    }

    fclose(file);
    return batch;
}

// Free a batch
void free_cifar10_batch(CIFAR10_Batch* batch) {
    if (batch) {
        if (batch->images) {
            free(batch->images);
        }
        free(batch);
    }
}

// Create an empty dataset
CIFAR10_Dataset* create_dataset(int num_samples) {
    CIFAR10_Dataset* dataset = (CIFAR10_Dataset*)malloc(sizeof(CIFAR10_Dataset));
    if (!dataset) {
        fprintf(stderr, "Error: Memory allocation failed for dataset\n");
        return NULL;
    }

    dataset->num_samples = num_samples;
    dataset->images = (float*)malloc(num_samples * CIFAR10_IMAGE_SIZE * sizeof(float));
    dataset->labels = (uint8_t*)malloc(num_samples * sizeof(uint8_t));

    if (!dataset->images || !dataset->labels) {
        fprintf(stderr, "Error: Memory allocation failed for dataset arrays\n");
        if (dataset->images) free(dataset->images);
        if (dataset->labels) free(dataset->labels);
        free(dataset);
        return NULL;
    }

    return dataset;
}

// Free a dataset
void free_dataset(CIFAR10_Dataset* dataset) {
    if (dataset) {
        if (dataset->images) free(dataset->images);
        if (dataset->labels) free(dataset->labels);
        free(dataset);
    }
}

// Normalize images from batch into dataset
void normalize_images(CIFAR10_Dataset* dataset, CIFAR10_Batch* batch, int offset) {
    for (int i = 0; i < batch->num_images; i++) {
        dataset->labels[offset + i] = batch->images[i].label;
        
        for (int j = 0; j < CIFAR10_IMAGE_SIZE; j++) {
            // Normalize pixel values from [0, 255] to [0, 1]
            dataset->images[(offset + i) * CIFAR10_IMAGE_SIZE + j] = 
                batch->images[i].data[j] / 255.0f;
        }
    }
}

// Load training data (1 to 5 batches)
CIFAR10_Dataset* load_training_data(const char* data_dir, int num_batches) {
    if (num_batches < 1 || num_batches > 5) {
        fprintf(stderr, "Error: num_batches must be between 1 and 5\n");
        return NULL;
    }
    
    const int total_samples = num_batches * CIFAR10_BATCH_SIZE;
    
    CIFAR10_Dataset* dataset = create_dataset(total_samples);
    if (!dataset) {
        return NULL;
    }

    char filename[256];
    for (int b = 0; b < num_batches; b++) {
        snprintf(filename, sizeof(filename), "%s/data_batch_%d.bin", data_dir, b + 1);
        printf("Loading %s...\n", filename);
        
        CIFAR10_Batch* batch = load_cifar10_batch(filename);
        if (!batch) {
            fprintf(stderr, "Error: Failed to load batch %d\n", b + 1);
            free_dataset(dataset);
            return NULL;
        }

        normalize_images(dataset, batch, b * CIFAR10_BATCH_SIZE);
        free_cifar10_batch(batch);
    }

    printf("Loaded %d training images\n", total_samples);
    return dataset;
}

// Load test data (1 batch)
CIFAR10_Dataset* load_test_data(const char* data_dir) {
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/test_batch.bin", data_dir);
    printf("Loading %s...\n", filename);

    CIFAR10_Batch* batch = load_cifar10_batch(filename);
    if (!batch) {
        fprintf(stderr, "Error: Failed to load test batch\n");
        return NULL;
    }

    CIFAR10_Dataset* dataset = create_dataset(CIFAR10_BATCH_SIZE);
    if (!dataset) {
        free_cifar10_batch(batch);
        return NULL;
    }

    normalize_images(dataset, batch, 0);
    free_cifar10_batch(batch);

    printf("Loaded %d test images\n", CIFAR10_BATCH_SIZE);
    return dataset;
}
