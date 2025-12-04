#include "data_loader.h"
#include "cnn.h"
#include "forward.h"
#include "backward.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <sys/time.h>
#endif

// Utility function to get current time in seconds
double get_time() {
#ifdef _WIN32
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / frequency.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
#endif
}

// Calculate cross-entropy loss
float calculate_loss(float* output, uint8_t* labels, int batch_size, int num_classes) {
    float loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        int label = labels[b];
        float prob = output[b * num_classes + label];
        loss -= logf(prob + 1e-7f);  // Add small epsilon to avoid log(0)
    }
    return loss / batch_size;
}

// Calculate accuracy
float calculate_accuracy(float* output, uint8_t* labels, int batch_size, int num_classes) {
    int correct = 0;
    for (int b = 0; b < batch_size; b++) {
        // Find predicted class
        int pred_class = 0;
        float max_prob = output[b * num_classes];
        for (int i = 1; i < num_classes; i++) {
            if (output[b * num_classes + i] > max_prob) {
                max_prob = output[b * num_classes + i];
                pred_class = i;
            }
        }
        
        if (pred_class == labels[b]) {
            correct++;
        }
    }
    return (float)correct / batch_size;
}

// Print progress bar
void print_progress_bar(int current, int total, float loss, float acc) {
    const int bar_width = 40;
    float progress = (float)current / total;
    int filled = (int)(progress * bar_width);
    
    printf("\r  Progress: [");
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) printf("=");
        else if (i == filled) printf(">");
        else printf(" ");
    }
    printf("] %d/%d (%.1f%%) - Loss: %.4f, Acc: %.4f", 
           current, total, progress * 100, loss, acc);
    fflush(stdout);
}

// Train for one epoch
void train_epoch(CNN* cnn, CIFAR10_Dataset* dataset, float learning_rate, int batch_size) {
    int num_batches = dataset->num_samples / batch_size;
    float total_loss = 0.0f;
    float total_acc = 0.0f;
    
    double epoch_start = get_time();
    
    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        int offset = batch_idx * batch_size;
        float* batch_images = &dataset->images[offset * CIFAR10_IMAGE_SIZE];
        uint8_t* batch_labels = &dataset->labels[offset];
        
        // Forward pass
        forward_pass(cnn, batch_images);
        
        // Calculate loss and accuracy
        float loss = calculate_loss(cnn->output, batch_labels, batch_size, FC2_OUTPUT_SIZE);
        float acc = calculate_accuracy(cnn->output, batch_labels, batch_size, FC2_OUTPUT_SIZE);
        total_loss += loss;
        total_acc += acc;
        
        // Backward pass
        backward_pass(cnn, batch_images, batch_labels);
        
        // Update weights
        update_weights(cnn, learning_rate);
        
        // Update progress bar
        print_progress_bar(batch_idx + 1, num_batches, 
                          total_loss / (batch_idx + 1), 
                          total_acc / (batch_idx + 1));
    }
    
    printf("\n");  // New line after progress bar
    double epoch_time = get_time() - epoch_start;
    printf("  Epoch completed in %.2f seconds - Avg Loss: %.4f, Avg Acc: %.4f\n",
           epoch_time, total_loss / num_batches, total_acc / num_batches);
}

// Evaluate on test set
void evaluate(CNN* cnn, CIFAR10_Dataset* dataset, int batch_size) {
    int num_batches = dataset->num_samples / batch_size;
    float total_loss = 0.0f;
    float total_acc = 0.0f;
    
    printf("Evaluating on %d samples...\n", dataset->num_samples);
    
    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        int offset = batch_idx * batch_size;
        float* batch_images = &dataset->images[offset * CIFAR10_IMAGE_SIZE];
        uint8_t* batch_labels = &dataset->labels[offset];
        
        // Forward pass only
        forward_pass(cnn, batch_images);
        
        // Calculate loss and accuracy
        float loss = calculate_loss(cnn->output, batch_labels, batch_size, FC2_OUTPUT_SIZE);
        float acc = calculate_accuracy(cnn->output, batch_labels, batch_size, FC2_OUTPUT_SIZE);
        total_loss += loss;
        total_acc += acc;
    }
    
    printf("Test Loss: %.4f, Test Accuracy: %.4f\n", 
           total_loss / num_batches, total_acc / num_batches);
}

int main(int argc, char** argv) {
    // Configuration
    const char* data_dir = "../cifar-10-batches-bin";
    const int batch_size = 64;
    const int num_epochs = 10;
    const float learning_rate = 0.001f;
    
    printf("=== CIFAR-10 CNN Training (CPU Baseline) ===\n");
    printf("Batch size: %d\n", batch_size);
    printf("Learning rate: %.4f\n", learning_rate);
    printf("Number of epochs: %d\n\n", num_epochs);
    
    // Load data
    printf("Loading training data...\n");
    CIFAR10_Dataset* train_data = load_training_data(data_dir);
    if (!train_data) {
        fprintf(stderr, "Failed to load training data\n");
        return 1;
    }
    
    printf("Loading test data...\n");
    CIFAR10_Dataset* test_data = load_test_data(data_dir);
    if (!test_data) {
        fprintf(stderr, "Failed to load test data\n");
        free_dataset(train_data);
        return 1;
    }
    
    printf("\n");
    
    // Create CNN
    printf("Creating CNN model...\n");
    CNN* cnn = create_cnn(batch_size);
    if (!cnn) {
        fprintf(stderr, "Failed to create CNN\n");
        free_dataset(train_data);
        free_dataset(test_data);
        return 1;
    }
    
    // Initialize weights
    initialize_weights(cnn);
    printf("\n");
    
    // Print model architecture
    printf("Model Architecture:\n");
    printf("  Input: %dx%dx%d\n", INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS);
    printf("  Conv1: %d filters, %dx%d kernel -> %dx%dx%d\n", 
           CONV1_FILTERS, CONV1_KERNEL_SIZE, CONV1_KERNEL_SIZE,
           CONV1_OUTPUT_SIZE, CONV1_OUTPUT_SIZE, CONV1_FILTERS);
    printf("  Pool1: %dx%d -> %dx%dx%d\n",
           POOL1_SIZE, POOL1_SIZE,
           POOL1_OUTPUT_SIZE, POOL1_OUTPUT_SIZE, CONV1_FILTERS);
    printf("  Conv2: %d filters, %dx%d kernel -> %dx%dx%d\n",
           CONV2_FILTERS, CONV2_KERNEL_SIZE, CONV2_KERNEL_SIZE,
           CONV2_OUTPUT_SIZE, CONV2_OUTPUT_SIZE, CONV2_FILTERS);
    printf("  Pool2: %dx%d -> %dx%dx%d\n",
           POOL2_SIZE, POOL2_SIZE,
           POOL2_OUTPUT_SIZE, POOL2_OUTPUT_SIZE, CONV2_FILTERS);
    printf("  FC1: %d -> %d\n", FC1_INPUT_SIZE, FC1_OUTPUT_SIZE);
    printf("  FC2: %d -> %d (output)\n", FC2_INPUT_SIZE, FC2_OUTPUT_SIZE);
    printf("\n");
    
    // Training loop
    double total_training_time = 0.0;
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        printf("Epoch %d/%d:\n", epoch + 1, num_epochs);
        
        double epoch_start = get_time();
        train_epoch(cnn, train_data, learning_rate, batch_size);
        double epoch_time = get_time() - epoch_start;
        total_training_time += epoch_time;
        
        // Evaluate on test set
        evaluate(cnn, test_data, batch_size);
        printf("\n");
    }
    
    printf("=== Training Complete ===\n");
    printf("Total training time: %.2f seconds\n", total_training_time);
    printf("Average time per epoch: %.2f seconds\n", total_training_time / num_epochs);
    
    // Cleanup
    free_cnn(cnn);
    free_dataset(train_data);
    free_dataset(test_data);
    
    return 0;
}
