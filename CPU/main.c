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

// Calculate reconstruction loss (MSE)
float calculate_reconstruction_loss(float* reconstructed, float* original, int size) {
    float loss = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = original[i] - reconstructed[i];
        loss += diff * diff;
    }
    return loss / size;
}

// Save reconstructed images to binary file
void save_reconstructed_images(float* images, int num_images, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filename);
        return;
    }
    
    // Write number of images
    fwrite(&num_images, sizeof(int), 1, f);
    
    // Write image size
    int image_size = CIFAR10_IMAGE_SIZE;
    fwrite(&image_size, sizeof(int), 1, f);
    
    // Write all images
    fwrite(images, sizeof(float), num_images * CIFAR10_IMAGE_SIZE, f);
    
    fclose(f);
    printf("  Saved %d reconstructed images to %s\n", num_images, filename);
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

// Print progress bar with reconstruction loss
void print_progress_bar_with_recon(int current, int total, float loss, float acc, float recon_loss) {
    const int bar_width = 40;
    float progress = (float)current / total;
    int filled = (int)(progress * bar_width);
    
    printf("\r  Progress: [");
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) printf("=");
        else if (i == filled) printf(">");
        else printf(" ");
    }
    printf("] %d/%d (%.1f%%) - Class Loss: %.4f, Acc: %.4f, Recon Loss: %.6f", 
           current, total, progress * 100, loss, acc, recon_loss);
    fflush(stdout);
}

// Train for one epoch
void train_epoch(CNN* cnn, CIFAR10_Dataset* dataset, float learning_rate, int batch_size, 
                 int is_last_epoch, const char* output_dir) {
    int num_batches = dataset->num_samples / batch_size;
    float total_loss = 0.0f;
    float total_acc = 0.0f;
    float total_recon_loss = 0.0f;
    
    // Storage for reconstructed images if last epoch
    float* all_reconstructed = NULL;
    if (is_last_epoch && cnn->decoder) {
        all_reconstructed = (float*)malloc(dataset->num_samples * CIFAR10_IMAGE_SIZE * sizeof(float));
    }
    
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
        
        // Backward pass (classification)
        backward_pass(cnn, batch_images, batch_labels);
        
        // Update classifier weights only (FC1, FC2)
        update_classifier_weights(cnn, learning_rate);
        
        // Train decoder and backprop to encoder (every epoch)
        if (cnn->decoder) {
            // Decoder forward pass
            decoder_forward(cnn->decoder, cnn->pool2->output);
            
            // Calculate reconstruction loss
            float recon_loss = calculate_reconstruction_loss(cnn->decoder->reconstructed, 
                                                            batch_images, 
                                                            batch_size * CIFAR10_IMAGE_SIZE);
            total_recon_loss += recon_loss;
            
            // Compute reconstruction gradient (MSE gradient)
            float* recon_grad = (float*)malloc(batch_size * 3 * 32 * 32 * sizeof(float));
            for (int i = 0; i < batch_size * CIFAR10_IMAGE_SIZE; i++) {
                recon_grad[i] = 2.0f * (cnn->decoder->reconstructed[i] - batch_images[i]) / 
                               (batch_size * CIFAR10_IMAGE_SIZE);
            }
            
            // Decoder backward pass (returns gradients for pool2 output)
            decoder_backward(cnn->decoder, cnn->pool2->output, recon_grad);
            
            // Backpropagate reconstruction gradients through encoder
            // This accumulates gradients with those from classification loss
            backprop_reconstruction_to_encoder(cnn, batch_images, cnn->decoder->upsample1->input_gradients, batch_size);
            
            // Update decoder weights
            update_decoder_weights(cnn->decoder, learning_rate);
            
            // Save reconstructed images for this batch (only in last epoch)
            if (is_last_epoch && all_reconstructed) {
                memcpy(&all_reconstructed[offset * CIFAR10_IMAGE_SIZE], 
                       cnn->decoder->reconstructed, 
                       batch_size * CIFAR10_IMAGE_SIZE * sizeof(float));
            }
            
            free(recon_grad);
        }
        
        // Update encoder weights with accumulated gradients from both losses
        update_encoder_weights(cnn, learning_rate);
        
        // Update progress bar with both losses
        float avg_recon = (cnn->decoder) ? total_recon_loss / (batch_idx + 1) : 0.0f;
        print_progress_bar_with_recon(batch_idx + 1, num_batches, 
                                      total_loss / (batch_idx + 1), 
                                      total_acc / (batch_idx + 1),
                                      avg_recon);
    }
    
    printf("\n");  // New line after progress bar
    double epoch_time = get_time() - epoch_start;
    
    // Always show reconstruction loss when decoder exists
    if (cnn->decoder) {
        printf("  Epoch completed in %.2f seconds - Class Loss: %.4f, Acc: %.4f, Recon Loss: %.6f\n",
               epoch_time, total_loss / num_batches, total_acc / num_batches, total_recon_loss / num_batches);
        
        // Save reconstructed images (only in last epoch)
        if (is_last_epoch && all_reconstructed) {
            char filename[256];
            snprintf(filename, sizeof(filename), "%s/reconstructed_images_cpu.bin", output_dir);
            save_reconstructed_images(all_reconstructed, dataset->num_samples, filename);
            free(all_reconstructed);
            printf("  Saved reconstructed images to %s\n", filename);
        }
    } else {
        printf("  Epoch completed in %.2f seconds - Avg Loss: %.4f, Avg Acc: %.4f\n",
               epoch_time, total_loss / num_batches, total_acc / num_batches);
    }
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
    const int batch_size = 32;
    int num_epochs = 20;  // Default value
    const float learning_rate = 0.01f;
    
    // Parse command-line arguments
    if (argc > 1) {
        num_epochs = atoi(argv[1]);
        if (num_epochs <= 0) {
            fprintf(stderr, "Invalid number of epochs: %s\n", argv[1]);
            fprintf(stderr, "Usage: %s [num_epochs]\n", argv[0]);
            return 1;
        }
    }
    
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
    
    // Create decoder for reconstruction training (trains all epochs)
    printf("Creating decoder for image reconstruction...\n");
    cnn->decoder = create_decoder(batch_size);
    if (!cnn->decoder) {
        fprintf(stderr, "Failed to create decoder\n");
        free_cnn(cnn);
        free_dataset(train_data);
        free_dataset(test_data);
        return 1;
    }
    initialize_decoder_weights(cnn->decoder);
    printf("Decoder initialized\n\n");
    
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
    printf("  Decoder: Pool2 -> Upsample -> TransConv(64ch) -> Upsample -> TransConv(3ch) -> 32x32x3\n");
    printf("\n");
    
    // Training loop
    double total_training_time = 0.0;
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        printf("Epoch %d/%d:\n", epoch + 1, num_epochs);
        
        // Check if this is the last epoch (for saving reconstructed images)
        int is_last_epoch = (epoch == num_epochs - 1);
        
        double epoch_start = get_time();
        train_epoch(cnn, train_data, learning_rate, batch_size, is_last_epoch, "../extracted_features");
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
