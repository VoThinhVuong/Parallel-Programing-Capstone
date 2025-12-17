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

// Compute reconstruction loss sum (MSE numerator) on device.
// Two-pass reduction to avoid heavy contention on a single atomicAdd.
__global__ void reconstruction_loss_partial_sum_kernel(const float* original, const float* reconstructed,
                                                       float* partial_sums, int total_size) {
    float thread_sum = 0.0f;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_size;
         idx += gridDim.x * blockDim.x) {
        float diff = original[idx] - reconstructed[idx];
        thread_sum += diff * diff;
    }

    extern __shared__ float s[];
    s[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s[threadIdx.x] += s[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = s[0];
    }
}

__global__ void reduce_sum_kernel(const float* partial_sums, float* out_sum, int n) {
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        thread_sum += partial_sums[i];
    }

    extern __shared__ float s[];
    s[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s[threadIdx.x] += s[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *out_sum = s[0];
    }
}

// Compute classification metrics on device: sum of cross-entropy losses and count of correct predictions.
// Writes one partial sum per block for both loss and correct.
__global__ void classification_metrics_partial_kernel(const float* probs, const uint8_t* labels,
                                                     float* partial_loss_sums, float* partial_correct_sums,
                                                     int batch_size, int num_classes) {
    float loss_sum = 0.0f;
    float correct_sum = 0.0f;

    for (int b = blockIdx.x * blockDim.x + threadIdx.x;
         b < batch_size;
         b += gridDim.x * blockDim.x) {
        int label = (int)labels[b];
        float prob = probs[b * num_classes + label];
        loss_sum += -logf(prob + 1e-7f);

        int pred_class = 0;
        float max_prob = probs[b * num_classes];
        for (int c = 1; c < num_classes; c++) {
            float p = probs[b * num_classes + c];
            if (p > max_prob) {
                max_prob = p;
                pred_class = c;
            }
        }
        if (pred_class == label) {
            correct_sum += 1.0f;
        }
    }

    extern __shared__ float s[];
    float* s_loss = s;
    float* s_correct = s + blockDim.x;
    s_loss[threadIdx.x] = loss_sum;
    s_correct[threadIdx.x] = correct_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_loss[threadIdx.x] += s_loss[threadIdx.x + stride];
            s_correct[threadIdx.x] += s_correct[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_loss_sums[blockIdx.x] = s_loss[0];
        partial_correct_sums[blockIdx.x] = s_correct[0];
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

// Train for one epoch on GPU (pre-allocated buffers passed in to avoid per-epoch overhead)
void train_epoch(CNN* cnn, CIFAR10_Dataset* dataset, float learning_rate, int batch_size,
                float* d_batch_images, uint8_t* d_batch_labels,
                float* d_class_loss_sum, float* d_class_correct_sum,
                float* d_class_loss_partial, float* d_class_correct_partial,
                float* d_recon_gradient, float* d_recon_loss_sum,
                float* d_recon_loss_partial, float* d_pool2_recon_gradient) {
    int num_batches = dataset->num_samples / batch_size;
    float total_loss = 0.0f;
    float total_acc = 0.0f;
    float total_recon_loss = 0.0f;
    float h_class_loss_sum = 0.0f;
    float h_class_correct_sum = 0.0f;
    float h_recon_loss_sum = 0.0f;
    
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

        // Classification loss/accuracy on device
        float loss = 0.0f;
        float acc = 0.0f;
        {
            int threads = 256;
            int blocks = (batch_size + threads - 1) / threads;
            if (blocks < 1) blocks = 1;

            classification_metrics_partial_kernel<<<blocks, threads, 2 * threads * sizeof(float)>>>(
                cnn->d_output, d_batch_labels,
                d_class_loss_partial, d_class_correct_partial,
                batch_size, FC2_OUTPUT_SIZE);
            CUDA_CHECK(cudaGetLastError());

            int reduce_threads = 256;
            reduce_sum_kernel<<<1, reduce_threads, reduce_threads * sizeof(float)>>>(
                d_class_loss_partial, d_class_loss_sum, blocks);
            reduce_sum_kernel<<<1, reduce_threads, reduce_threads * sizeof(float)>>>(
                d_class_correct_partial, d_class_correct_sum, blocks);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpy(&h_class_loss_sum, d_class_loss_sum, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_class_correct_sum, d_class_correct_sum, sizeof(float), cudaMemcpyDeviceToHost));

            loss = h_class_loss_sum / batch_size;
            acc = h_class_correct_sum / batch_size;
        }
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
            {
                int threads = 256;
                int total_size = batch_size * CIFAR10_IMAGE_SIZE;
                int blocks = (total_size + threads - 1) / threads;
                reconstruction_loss_partial_sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(
                    d_batch_images, cnn->decoder->d_reconstructed, d_recon_loss_partial, total_size);
                CUDA_CHECK(cudaGetLastError());

                int reduce_threads = 256;
                reduce_sum_kernel<<<1, reduce_threads, reduce_threads * sizeof(float)>>>(
                    d_recon_loss_partial, d_recon_loss_sum, blocks);
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
}

// Evaluate on test set (pre-allocated buffers passed in)
void evaluate(CNN* cnn, CIFAR10_Dataset* dataset, int batch_size,
             float* d_batch_images, uint8_t* d_batch_labels,
             float* d_class_loss_sum, float* d_class_correct_sum,
             float* d_class_loss_partial, float* d_class_correct_partial) {
    int num_batches = dataset->num_samples / batch_size;
    float total_loss = 0.0f;
    float total_acc = 0.0f;
    float h_class_loss_sum = 0.0f;
    float h_class_correct_sum = 0.0f;
    
    printf("Evaluating on %d samples...\n", dataset->num_samples);
    
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
        
        // Forward pass only
        forward_pass(cnn, d_batch_images);

        float loss = 0.0f;
        float acc = 0.0f;
        {
            int threads = 256;
            int blocks = (batch_size + threads - 1) / threads;
            if (blocks < 1) blocks = 1;

            classification_metrics_partial_kernel<<<blocks, threads, 2 * threads * sizeof(float)>>>(
                cnn->d_output, d_batch_labels,
                d_class_loss_partial, d_class_correct_partial,
                batch_size, FC2_OUTPUT_SIZE);
            CUDA_CHECK(cudaGetLastError());

            int reduce_threads = 256;
            reduce_sum_kernel<<<1, reduce_threads, reduce_threads * sizeof(float)>>>(
                d_class_loss_partial, d_class_loss_sum, blocks);
            reduce_sum_kernel<<<1, reduce_threads, reduce_threads * sizeof(float)>>>(
                d_class_correct_partial, d_class_correct_sum, blocks);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpy(&h_class_loss_sum, d_class_loss_sum, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_class_correct_sum, d_class_correct_sum, sizeof(float), cudaMemcpyDeviceToHost));

            loss = h_class_loss_sum / batch_size;
            acc = h_class_correct_sum / batch_size;
        }
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
    
    printf("=== CIFAR-10 CNN Training (GPU v4 - Further Optimized) ===\n");
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
    
    // Allocate metric buffers once for all epochs (avoid per-epoch malloc/free overhead)
    float* d_batch_images;
    uint8_t* d_batch_labels;
    float* d_class_loss_sum;
    float* d_class_correct_sum;
    float* d_class_loss_partial;
    float* d_class_correct_partial;
    float* d_recon_gradient = NULL;
    float* d_recon_loss_sum = NULL;
    float* d_recon_loss_partial = NULL;
    float* d_pool2_recon_gradient = NULL;

    CUDA_CHECK(cudaMalloc(&d_batch_images, batch_size * CIFAR10_IMAGE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_batch_labels, batch_size * sizeof(uint8_t)));

    {
        int threads = 256;
        int blocks = (batch_size + threads - 1) / threads;
        if (blocks < 1) blocks = 1;
        CUDA_CHECK(cudaMalloc(&d_class_loss_sum, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_class_correct_sum, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_class_loss_partial, blocks * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_class_correct_partial, blocks * sizeof(float)));
    }

    if (cnn->decoder) {
        CUDA_CHECK(cudaMalloc(&d_recon_gradient, batch_size * CIFAR10_IMAGE_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_recon_loss_sum, sizeof(float)));
        d_pool2_recon_gradient = cnn->decoder->upsample1->d_input_gradients;
        int total_size = batch_size * CIFAR10_IMAGE_SIZE;
        int threads = 256;
        int blocks = (total_size + threads - 1) / threads;
        CUDA_CHECK(cudaMalloc(&d_recon_loss_partial, blocks * sizeof(float)));
    }

    // Training loop
    double total_training_time = 0.0;
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        printf("Epoch %d/%d:\n", epoch + 1, num_epochs);
        
        double epoch_start = get_time();
        train_epoch(cnn, train_data, learning_rate, batch_size,
                   d_batch_images, d_batch_labels,
                   d_class_loss_sum, d_class_correct_sum, d_class_loss_partial, d_class_correct_partial,
                   d_recon_gradient, d_recon_loss_sum, d_recon_loss_partial, d_pool2_recon_gradient);
        double epoch_time = get_time() - epoch_start;
        total_training_time += epoch_time;
        
        // Evaluate on test set
        evaluate(cnn, test_data, batch_size,
                d_batch_images, d_batch_labels,
                d_class_loss_sum, d_class_correct_sum, d_class_loss_partial, d_class_correct_partial);
        printf("\n");
    }

    // Free shared metric buffers
    CUDA_CHECK(cudaFree(d_batch_images));
    CUDA_CHECK(cudaFree(d_batch_labels));
    CUDA_CHECK(cudaFree(d_class_loss_sum));
    CUDA_CHECK(cudaFree(d_class_correct_sum));
    CUDA_CHECK(cudaFree(d_class_loss_partial));
    CUDA_CHECK(cudaFree(d_class_correct_partial));
    if (d_recon_gradient) CUDA_CHECK(cudaFree(d_recon_gradient));
    if (d_recon_loss_sum) CUDA_CHECK(cudaFree(d_recon_loss_sum));
    if (d_recon_loss_partial) CUDA_CHECK(cudaFree(d_recon_loss_partial));
    
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
