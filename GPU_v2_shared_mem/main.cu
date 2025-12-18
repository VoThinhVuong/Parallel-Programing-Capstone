#include "data_loader.h"
#include "cnn.cuh"
#include "forward.cuh"
#include "backward.cuh"
#include "feature_extractor.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifdef _WIN32
    #include <windows.h>
    #include <direct.h>  // For _mkdir
#else
    #include <sys/time.h>
    #include <sys/stat.h>  // For mkdir
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

// Calculate cross-entropy loss (on host, data copied from device)
float calculate_loss(float* output, uint8_t* labels, int batch_size, int num_classes) {
    float loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        int label = labels[b];
        float prob = output[b * num_classes + label];
        loss -= logf(prob + 1e-7f);  // Add small epsilon to avoid log(0)
    }
    return loss / batch_size;
}

// Calculate accuracy (on host, data copied from device)
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

// Calculate reconstruction loss (MSE) on host
float calculate_reconstruction_loss(float* reconstructed, float* original, int size) {
    float loss = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = original[i] - reconstructed[i];
        loss += diff * diff;
    }
    return loss / size;
}

// Compute reconstruction gradient (MSE gradient: 2 * (reconstructed - original) / N)
__global__ void compute_reconstruction_gradient_kernel(float* original, float* reconstructed,
                                                      float* gradient, int batch_size, int image_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * image_size;
    if (idx < total_size) {
        // Weight reconstruction loss lower than classification
        float scale = 0.01f;
        gradient[idx] = scale * 2.0f * (reconstructed[idx] - original[idx]) / total_size;
    }
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

// Train for one epoch on GPU
void train_epoch(CNN* cnn, CIFAR10_Dataset* dataset, float learning_rate, int batch_size,
                 int is_last_epoch, const char* output_dir) {
    int num_batches = dataset->num_samples / batch_size;
    float total_loss = 0.0f;
    float total_acc = 0.0f;
    float total_recon_loss = 0.0f;
    
    // Allocate device memory for batch data
    float* d_batch_images;
    uint8_t* d_batch_labels;
    CUDA_CHECK(cudaMalloc(&d_batch_images, batch_size * CIFAR10_IMAGE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_batch_labels, batch_size * sizeof(uint8_t)));
    
    // Allocate host memory for output (for loss/accuracy calculation)
    float* h_output = (float*)malloc(batch_size * FC2_OUTPUT_SIZE * sizeof(float));

    // Allocate for reconstruction if decoder exists
    float* h_reconstructed = NULL;
    float* d_recon_grad = NULL;
    float* all_reconstructed = NULL;
    if (cnn->decoder != NULL) {
        h_reconstructed = (float*)malloc(batch_size * CIFAR10_IMAGE_SIZE * sizeof(float));
        CUDA_CHECK(cudaMalloc(&d_recon_grad, batch_size * CIFAR10_IMAGE_SIZE * sizeof(float)));
        
        // Allocate storage for all reconstructed images if last epoch
        if (is_last_epoch) {
            all_reconstructed = (float*)malloc(dataset->num_samples * CIFAR10_IMAGE_SIZE * sizeof(float));
        }
    }
    
    double epoch_start = get_time();
    
    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        int offset = batch_idx * batch_size;
        float* batch_images = &dataset->images[offset * CIFAR10_IMAGE_SIZE];
        uint8_t* batch_labels = &dataset->labels[offset];
        
        // Copy batch to device
        CUDA_CHECK(cudaMemcpy(d_batch_images, batch_images, 
                             batch_size * CIFAR10_IMAGE_SIZE * sizeof(float), 
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_batch_labels, batch_labels, 
                             batch_size * sizeof(uint8_t), 
                             cudaMemcpyHostToDevice));
        
        // Forward pass
        forward_pass(cnn, d_batch_images);
        
        // Copy output back to host for metrics
        CUDA_CHECK(cudaMemcpy(h_output, cnn->d_output, 
                             batch_size * FC2_OUTPUT_SIZE * sizeof(float), 
                             cudaMemcpyDeviceToHost));
        
        // Calculate loss and accuracy
        float loss = calculate_loss(h_output, batch_labels, batch_size, FC2_OUTPUT_SIZE);
        float acc = calculate_accuracy(h_output, batch_labels, batch_size, FC2_OUTPUT_SIZE);
        total_loss += loss;
        total_acc += acc;
        
        // Backward pass for classification
        backward_pass(cnn, d_batch_images, d_batch_labels);

        // Update classifier weights (FC layers only)
        update_classifier_weights(cnn, learning_rate);

        // Decoder training (if decoder exists)
        float recon_loss = 0.0f;
        if (cnn->decoder != NULL) {
            // Forward pass through decoder
            decoder_forward(cnn->decoder, cnn->pool2->d_output, batch_size);

            // Copy reconstructed images to host for loss calculation
            CUDA_CHECK(cudaMemcpy(h_reconstructed, cnn->decoder->d_reconstructed,
                                  batch_size * CIFAR10_IMAGE_SIZE * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            // Calculate reconstruction loss (MSE)
            recon_loss = calculate_reconstruction_loss(h_reconstructed, batch_images,
                                                       batch_size * CIFAR10_IMAGE_SIZE);
            total_recon_loss += recon_loss;

            // Compute gradient of reconstruction loss on device
            int threads = 256;
            int total = batch_size * CIFAR10_IMAGE_SIZE;
            int blocks = (total + threads - 1) / threads;
            compute_reconstruction_gradient_kernel<<<blocks, threads>>>(
                d_batch_images, cnn->decoder->d_reconstructed, d_recon_grad,
                batch_size, CIFAR10_IMAGE_SIZE);
            CUDA_CHECK(cudaGetLastError());

            // Backward pass through decoder
            decoder_backward(cnn->decoder, d_recon_grad, cnn->pool2->d_output, batch_size);

            // Backpropagate reconstruction gradients through encoder
            backprop_reconstruction_to_encoder(cnn, d_batch_images,
                                               cnn->decoder->upsample1->d_input_gradients,
                                               batch_size);

            // Update decoder weights
            update_decoder_weights(cnn->decoder, learning_rate);
            
            // Accumulate reconstructed images for this batch (only in last epoch)
            if (is_last_epoch && all_reconstructed) {
                memcpy(&all_reconstructed[offset * CIFAR10_IMAGE_SIZE], 
                       h_reconstructed, 
                       batch_size * CIFAR10_IMAGE_SIZE * sizeof(float));
            }
        }

        // Update encoder weights (with accumulated gradients from both losses)
        update_encoder_weights(cnn, learning_rate);
        
        // Update progress bar
        if (cnn->decoder != NULL) {
            print_progress_bar_with_recon(batch_idx + 1, num_batches,
                                          total_loss / (batch_idx + 1),
                                          total_acc / (batch_idx + 1),
                                          total_recon_loss / (batch_idx + 1));
        } else {
            print_progress_bar(batch_idx + 1, num_batches,
                               total_loss / (batch_idx + 1),
                               total_acc / (batch_idx + 1));
        }
    }
    
    printf("\n");  // New line after progress bar
    double epoch_time = get_time() - epoch_start;
    if (cnn->decoder != NULL) {
        printf("  Epoch completed in %.2f seconds - Avg Class Loss: %.4f, Avg Acc: %.4f, Avg Recon Loss: %.6f\n",
               epoch_time, total_loss / num_batches, total_acc / num_batches,
               total_recon_loss / num_batches);
        
        // Save reconstructed images (only in last epoch)
        if (is_last_epoch && all_reconstructed) {
            char filename[256];
            snprintf(filename, sizeof(filename), "%s/reconstructed_images_gpu_v2.bin", output_dir);
            save_reconstructed_images(all_reconstructed, dataset->num_samples, filename);
        }
    } else {
        printf("  Epoch completed in %.2f seconds - Avg Loss: %.4f, Avg Acc: %.4f\n",
               epoch_time, total_loss / num_batches, total_acc / num_batches);
    }
    
    // Cleanup
    free(h_output);
    if (h_reconstructed) free(h_reconstructed);
    if (all_reconstructed) free(all_reconstructed);
    if (d_recon_grad) CUDA_CHECK(cudaFree(d_recon_grad));
    CUDA_CHECK(cudaFree(d_batch_images));
    CUDA_CHECK(cudaFree(d_batch_labels));
}

// Evaluate on test set
void evaluate(CNN* cnn, CIFAR10_Dataset* dataset, int batch_size) {
    int num_batches = dataset->num_samples / batch_size;
    float total_loss = 0.0f;
    float total_acc = 0.0f;
    
    // Allocate device memory for batch data
    float* d_batch_images;
    uint8_t* d_batch_labels;
    CUDA_CHECK(cudaMalloc(&d_batch_images, batch_size * CIFAR10_IMAGE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_batch_labels, batch_size * sizeof(uint8_t)));
    
    // Allocate host memory for output
    float* h_output = (float*)malloc(batch_size * FC2_OUTPUT_SIZE * sizeof(float));
    
    printf("Evaluating on %d samples...\n", dataset->num_samples);
    
    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        int offset = batch_idx * batch_size;
        float* batch_images = &dataset->images[offset * CIFAR10_IMAGE_SIZE];
        uint8_t* batch_labels = &dataset->labels[offset];
        
        // Copy batch to device
        CUDA_CHECK(cudaMemcpy(d_batch_images, batch_images, 
                             batch_size * CIFAR10_IMAGE_SIZE * sizeof(float), 
                             cudaMemcpyHostToDevice));
        
        // Forward pass only
        forward_pass(cnn, d_batch_images);
        
        // Copy output back to host
        CUDA_CHECK(cudaMemcpy(h_output, cnn->d_output, 
                             batch_size * FC2_OUTPUT_SIZE * sizeof(float), 
                             cudaMemcpyDeviceToHost));
        
        // Calculate loss and accuracy
        float loss = calculate_loss(h_output, batch_labels, batch_size, FC2_OUTPUT_SIZE);
        float acc = calculate_accuracy(h_output, batch_labels, batch_size, FC2_OUTPUT_SIZE);
        total_loss += loss;
        total_acc += acc;
    }
    
    printf("Test Loss: %.4f, Test Accuracy: %.4f\n", 
           total_loss / num_batches, total_acc / num_batches);
    
    // Cleanup
    free(h_output);
    CUDA_CHECK(cudaFree(d_batch_images));
    CUDA_CHECK(cudaFree(d_batch_labels));
}

int main(int argc, char** argv) {
    // Configuration
    const char* data_dir = "../cifar-10-batches-bin";
    const int batch_size = 64;
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
    
    printf("=== CIFAR-10 CNN Training (GPU Shared Memory Implementation) ===\n");
    printf("Batch size: %d\n", batch_size);
    printf("Learning rate: %.4f\n", learning_rate);
    printf("Number of epochs: %d\n\n", num_epochs);
    
    // Print GPU information
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory: %.2f GB\n\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    
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
    printf("Creating CNN model on GPU...\n");
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

    // Create decoder
    printf("Creating decoder...\n");
    cnn->decoder = create_decoder(batch_size);
    if (!cnn->decoder) {
        fprintf(stderr, "Failed to create decoder\n");
        free_cnn(cnn);
        free_dataset(train_data);
        free_dataset(test_data);
        return 1;
    }
    initialize_decoder_weights(cnn->decoder);
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
    printf("  Decoder:\n");
    printf("    Input: %dx%dx%d (from Pool2)\n", POOL2_OUTPUT_SIZE, POOL2_OUTPUT_SIZE, CONV2_FILTERS);
    printf("    Upsample1: -> %dx%d\n", 16, 16);
    printf("    TransConv1: -> %dx%dx%d\n", 16, 16, 64);
    printf("    Upsample2: -> %dx%d\n", 32, 32);
    printf("    TransConv2: -> %dx%dx%d (reconstructed)\n", 32, 32, 3);
    printf("\n");

    // Ensure output dir exists (for consistency with other versions)
    const char* output_dir = "../extracted_features";
    #ifdef _WIN32
        _mkdir(output_dir);
    #else
        mkdir(output_dir, 0777);
    #endif
    
    // Training loop
    double total_training_time = 0.0;
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        printf("Epoch %d/%d:\n", epoch + 1, num_epochs);
        
        double epoch_start = get_time();
        int is_last_epoch = (epoch == num_epochs - 1);
        train_epoch(cnn, train_data, learning_rate, batch_size, is_last_epoch, output_dir);
        double epoch_time = get_time() - epoch_start;
        total_training_time += epoch_time;
        
        // Evaluate on test set
        evaluate(cnn, test_data, batch_size);
        printf("\n");
    }
    
    printf("=== Training Complete ===\n");
    printf("Total training time: %.2f seconds\n", total_training_time);
    printf("Average time per epoch: %.2f seconds\n", total_training_time / num_epochs);
    
    // Save encoder weights for feature extraction
    printf("\nSaving encoder weights...\n");
    if (save_encoder_weights(cnn, "encoder_weights.bin") == 0) {
        printf("Encoder weights saved to 'encoder_weights.bin'\n");
        printf("You can now run './extract_features' to extract features and train a classifier\n");
    }
    
    // Cleanup
    free_cnn(cnn);
    free_dataset(train_data);
    free_dataset(test_data);
    
    CUDA_CHECK(cudaDeviceReset());
    
    return 0;
}
