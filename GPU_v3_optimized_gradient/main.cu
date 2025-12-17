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

// Compute reconstruction loss sum (MSE numerator) on device to avoid large D2H copies
__global__ void reconstruction_loss_sum_kernel(const float* original, const float* reconstructed,
                                               float* loss_sum, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        float diff = original[idx] - reconstructed[idx];
        atomicAdd(loss_sum, diff * diff);
    }
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
void train_epoch(CNN* cnn, CIFAR10_Dataset* dataset, float learning_rate, int batch_size) {
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
    float* d_recon_gradient = NULL;
    float* d_pool2_recon_gradient = NULL;
    float* d_recon_loss_sum = NULL;
    float h_recon_loss_sum = 0.0f;
    if (cnn->decoder) {
        CUDA_CHECK(cudaMalloc(&d_recon_gradient, batch_size * CIFAR10_IMAGE_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pool2_recon_gradient, batch_size * FC1_INPUT_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_recon_loss_sum, sizeof(float)));
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
        
        // Backward pass
        backward_pass(cnn, d_batch_images, d_batch_labels);

        if (cnn->decoder) {
            // Update classifier weights first (FC only)
            update_classifier_weights(cnn, learning_rate);

            // Decoder forward
            decoder_forward(cnn->decoder, cnn->pool2->d_output, batch_size);

            // Reconstruction loss (MSE) computed on device; copy only a scalar
            CUDA_CHECK(cudaMemset(d_recon_loss_sum, 0, sizeof(float)));
            {
                int threads = 256;
                int total_size = batch_size * CIFAR10_IMAGE_SIZE;
                int blocks = (total_size + threads - 1) / threads;
                reconstruction_loss_sum_kernel<<<blocks, threads>>>(
                    d_batch_images, cnn->decoder->d_reconstructed, d_recon_loss_sum, total_size);
                CUDA_CHECK(cudaGetLastError());
            }
            CUDA_CHECK(cudaMemcpy(&h_recon_loss_sum, d_recon_loss_sum, sizeof(float), cudaMemcpyDeviceToHost));
            float recon_loss = h_recon_loss_sum / (batch_size * CIFAR10_IMAGE_SIZE);
            total_recon_loss += recon_loss;

            // Compute gradient of reconstruction loss on device
            int threads = 256;
            int total_size = batch_size * CIFAR10_IMAGE_SIZE;
            int blocks = (total_size + threads - 1) / threads;
            compute_reconstruction_gradient_kernel<<<blocks, threads>>>(
                d_batch_images, cnn->decoder->d_reconstructed, d_recon_gradient,
                batch_size, CIFAR10_IMAGE_SIZE);
            CUDA_CHECK(cudaGetLastError());

            // Decoder backward: produce gradient w.r.t pool2 output
            decoder_backward(cnn->decoder, d_recon_gradient, d_pool2_recon_gradient, batch_size);

            // Backpropagate reconstruction gradients through encoder (accumulate into conv grads)
            backprop_reconstruction_to_encoder(cnn, d_batch_images, d_pool2_recon_gradient, batch_size);

            // Update decoder and encoder weights
            update_decoder_weights(cnn->decoder, learning_rate);
            update_encoder_weights(cnn, learning_rate);

            // Update progress bar
            print_progress_bar_with_recon(batch_idx + 1, num_batches,
                                          total_loss / (batch_idx + 1),
                                          total_acc / (batch_idx + 1),
                                          total_recon_loss / (batch_idx + 1));
        } else {
            // Update weights
            update_weights(cnn, learning_rate);

            // Update progress bar
            print_progress_bar(batch_idx + 1, num_batches,
                              total_loss / (batch_idx + 1),
                              total_acc / (batch_idx + 1));
        }
    }
    
    printf("\n");  // New line after progress bar
    double epoch_time = get_time() - epoch_start;
    printf("  Epoch completed in %.2f seconds - Avg Loss: %.4f, Avg Acc: %.4f\n",
           epoch_time, total_loss / num_batches, total_acc / num_batches);
    
    // Cleanup
    free(h_output);
    CUDA_CHECK(cudaFree(d_batch_images));
    CUDA_CHECK(cudaFree(d_batch_labels));
    if (d_recon_gradient) CUDA_CHECK(cudaFree(d_recon_gradient));
    if (d_pool2_recon_gradient) CUDA_CHECK(cudaFree(d_pool2_recon_gradient));
    if (d_recon_loss_sum) CUDA_CHECK(cudaFree(d_recon_loss_sum));
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
    
    printf("=== CIFAR-10 CNN Training (GPU v3 - Optimized Gradients) ===\n");
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

    // Create and initialize decoder
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
    printf("\n");
    // Print decoder architecture if present
    if (cnn->decoder) {
        printf("Decoder Architecture:\n");
        printf("  Input: %dx%dx%d (pool2 output)\n", POOL2_OUTPUT_SIZE, POOL2_OUTPUT_SIZE, CONV2_FILTERS);
        printf("  Upsample1: %dx%d -> %dx%d (scale 2)\n", POOL2_OUTPUT_SIZE, POOL2_OUTPUT_SIZE, POOL2_OUTPUT_SIZE*2, POOL2_OUTPUT_SIZE*2);
        printf("  TransposeConv1: %d -> %d, kernel %d\n", CONV2_FILTERS, 64, 3);
        printf("  Upsample2: %dx%d -> %dx%d (scale 2)\n", POOL2_OUTPUT_SIZE*2, POOL2_OUTPUT_SIZE*2, POOL2_OUTPUT_SIZE*4, POOL2_OUTPUT_SIZE*4);
        printf("  TransposeConv2: %d -> %d, kernel %d (reconstructed image)\n", 64, 3, 3);
        printf("\n");
    }
    
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
