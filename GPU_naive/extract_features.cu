#include "feature_extractor.cuh"
#include "data_loader.h"
#include "cnn.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <sys/time.h>
#endif

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

// Simple logistic regression classifier
typedef struct {
    float* weights;  // feature_size x num_classes
    float* bias;     // num_classes
    int feature_size;
    int num_classes;
} LogisticRegression;

LogisticRegression* create_classifier(int feature_size, int num_classes) {
    LogisticRegression* clf = (LogisticRegression*)malloc(sizeof(LogisticRegression));
    clf->feature_size = feature_size;
    clf->num_classes = num_classes;
    
    clf->weights = (float*)calloc(feature_size * num_classes, sizeof(float));
    clf->bias = (float*)calloc(num_classes, sizeof(float));
    
    // Xavier initialization
    float scale = sqrtf(2.0f / (feature_size + num_classes));
    for (int i = 0; i < feature_size * num_classes; i++) {
        clf->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
    
    return clf;
}

void free_classifier(LogisticRegression* clf) {
    if (clf) {
        if (clf->weights) free(clf->weights);
        if (clf->bias) free(clf->bias);
        free(clf);
    }
}

void softmax(float* logits, float* output, int num_classes) {
    float max_logit = logits[0];
    for (int i = 1; i < num_classes; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        output[i] = expf(logits[i] - max_logit);
        sum += output[i];
    }
    
    for (int i = 0; i < num_classes; i++) {
        output[i] /= sum;
    }
}

float predict_and_evaluate(LogisticRegression* clf, float* features, uint8_t* labels, 
                           int num_samples) {
    int correct = 0;
    float* logits = (float*)malloc(clf->num_classes * sizeof(float));
    float* probs = (float*)malloc(clf->num_classes * sizeof(float));
    
    for (int n = 0; n < num_samples; n++) {
        // Compute logits
        for (int c = 0; c < clf->num_classes; c++) {
            logits[c] = clf->bias[c];
            for (int f = 0; f < clf->feature_size; f++) {
                logits[c] += features[n * clf->feature_size + f] * 
                            clf->weights[f * clf->num_classes + c];
            }
        }
        
        // Softmax
        softmax(logits, probs, clf->num_classes);
        
        // Predict
        int pred = 0;
        float max_prob = probs[0];
        for (int c = 1; c < clf->num_classes; c++) {
            if (probs[c] > max_prob) {
                max_prob = probs[c];
                pred = c;
            }
        }
        
        if (pred == labels[n]) correct++;
    }
    
    free(logits);
    free(probs);
    
    return (float)correct / num_samples;
}

void train_classifier(LogisticRegression* clf, float* features, uint8_t* labels,
                     int num_samples, int num_epochs, float learning_rate) {
    printf("\nTraining classifier...\n");
    printf("Samples: %d, Features: %d, Classes: %d\n", 
           num_samples, clf->feature_size, clf->num_classes);
    printf("Epochs: %d, Learning rate: %.4f\n\n", num_epochs, learning_rate);
    
    float* logits = (float*)malloc(clf->num_classes * sizeof(float));
    float* probs = (float*)malloc(clf->num_classes * sizeof(float));
    float* gradients = (float*)malloc(clf->num_classes * sizeof(float));
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        double epoch_start = get_time();
        float total_loss = 0.0f;
        
        for (int n = 0; n < num_samples; n++) {
            // Forward pass
            for (int c = 0; c < clf->num_classes; c++) {
                logits[c] = clf->bias[c];
                for (int f = 0; f < clf->feature_size; f++) {
                    logits[c] += features[n * clf->feature_size + f] * 
                                clf->weights[f * clf->num_classes + c];
                }
            }
            
            softmax(logits, probs, clf->num_classes);
            
            // Compute loss
            total_loss -= logf(probs[labels[n]] + 1e-7f);
            
            // Compute gradients
            for (int c = 0; c < clf->num_classes; c++) {
                gradients[c] = probs[c];
                if (c == labels[n]) gradients[c] -= 1.0f;
            }
            
            // Update weights and bias
            for (int c = 0; c < clf->num_classes; c++) {
                clf->bias[c] -= learning_rate * gradients[c];
                for (int f = 0; f < clf->feature_size; f++) {
                    clf->weights[f * clf->num_classes + c] -= 
                        learning_rate * gradients[c] * features[n * clf->feature_size + f];
                }
            }
        }
        
        double epoch_time = get_time() - epoch_start;
        float avg_loss = total_loss / num_samples;
        
        printf("Epoch %d/%d - Loss: %.4f - Time: %.2fs\n", 
               epoch + 1, num_epochs, avg_loss, epoch_time);
    }
    
    free(logits);
    free(probs);
    free(gradients);
}

int main(int argc, char** argv) {
    const char* data_dir = "../cifar-10-batches-bin";
    const char* encoder_weights_file = "encoder_weights.bin";
    const char* train_features_file = "../extracted_features/train_features_naive.bin";
    const char* test_features_file = "../extracted_features/test_features_naive.bin";
    const int batch_size = 64;
    
    printf("=== CIFAR-10 Feature Extraction & Classification ===\n\n");
    
    // Check if we need to extract features or can load them
    bool extract_train = true;
    bool extract_test = true;
    
    FILE* test_file = fopen(train_features_file, "rb");
    if (test_file) {
        fclose(test_file);
        printf("Found existing training features file.\n");
        char response;
        printf("Re-extract features? (y/n): ");
        scanf(" %c", &response);
        extract_train = (response == 'y' || response == 'Y');
    }
    
    test_file = fopen(test_features_file, "rb");
    if (test_file) {
        fclose(test_file);
        extract_test = extract_train;  // Extract both or neither
    }
    
    float* train_features = NULL;
    float* test_features = NULL;
    int train_num_samples = 0, train_feature_size = 0;
    int test_num_samples = 0, test_feature_size = 0;
    
    // Load or extract features
    if (extract_train || extract_test) {
        // Load datasets
        printf("Loading training data...\n");
        CIFAR10_Dataset* train_data = load_training_data(data_dir, 5);
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
        
        // Create CNN and load encoder weights
        printf("\nCreating CNN model...\n");
        CNN* cnn = create_cnn(batch_size);
        if (!cnn) {
            fprintf(stderr, "Failed to create CNN\n");
            free_dataset(train_data);
            free_dataset(test_data);
            return 1;
        }
        
        // Load trained encoder weights
        if (load_encoder_weights(cnn, encoder_weights_file) != 0) {
            fprintf(stderr, "Warning: Could not load encoder weights. Using random initialization.\n");
            fprintf(stderr, "Please train the model first using main program.\n");
            free_cnn(cnn);
            free_dataset(train_data);
            free_dataset(test_data);
            return 1;
        }
        
        // Extract features
        if (extract_train) {
            printf("\n=== Extracting Training Features ===\n");
            train_features = extract_features(cnn, train_data, batch_size);
            if (!train_features) {
                fprintf(stderr, "Failed to extract training features\n");
                free_cnn(cnn);
                free_dataset(train_data);
                free_dataset(test_data);
                return 1;
            }
            save_features(train_features_file, train_features, train_data->num_samples, FEATURE_SIZE);
            train_num_samples = train_data->num_samples;
            train_feature_size = FEATURE_SIZE;
        }
        
        if (extract_test) {
            printf("\n=== Extracting Test Features ===\n");
            test_features = extract_features(cnn, test_data, batch_size);
            if (!test_features) {
                fprintf(stderr, "Failed to extract test features\n");
                if (train_features) free(train_features);
                free_cnn(cnn);
                free_dataset(train_data);
                free_dataset(test_data);
                return 1;
            }
            save_features(test_features_file, test_features, test_data->num_samples, FEATURE_SIZE);
            test_num_samples = test_data->num_samples;
            test_feature_size = FEATURE_SIZE;
        }
        
        // Save labels too
        if (extract_train) {
            FILE* f = fopen("../extracted_features/train_labels.bin", "wb");
            if (f) {
                fwrite(&train_data->num_samples, sizeof(int), 1, f);
                fwrite(train_data->labels, sizeof(uint8_t), train_data->num_samples, f);
                fclose(f);
                printf("Saved training labels\n");
            }
        }
        
        if (extract_test) {
            FILE* f = fopen("../extracted_features/test_labels.bin", "wb");
            if (f) {
                fwrite(&test_data->num_samples, sizeof(int), 1, f);
                fwrite(test_data->labels, sizeof(uint8_t), test_data->num_samples, f);
                fclose(f);
                printf("Saved test labels\n");
            }
        }
        
        free_cnn(cnn);
        free_dataset(train_data);
        free_dataset(test_data);
        
        printf("\nFeature extraction complete!\n");
    }
    
    // Load features if not extracted
    if (!train_features) {
        train_features = load_features(train_features_file, &train_num_samples, &train_feature_size);
        if (!train_features) {
            fprintf(stderr, "Failed to load training features\n");
            return 1;
        }
    }
    
    if (!test_features) {
        test_features = load_features(test_features_file, &test_num_samples, &test_feature_size);
        if (!test_features) {
            fprintf(stderr, "Failed to load test features\n");
            free(train_features);
            return 1;
        }
    }
    
    // Load labels
    uint8_t* train_labels = NULL;
    uint8_t* test_labels = NULL;
    
    FILE* f = fopen("../extracted_features/train_labels.bin", "rb");
    if (f) {
        int n;
        fread(&n, sizeof(int), 1, f);
        train_labels = (uint8_t*)malloc(n * sizeof(uint8_t));
        fread(train_labels, sizeof(uint8_t), n, f);
        fclose(f);
    }
    
    f = fopen("../extracted_features/test_labels.bin", "rb");
    if (f) {
        int n;
        fread(&n, sizeof(int), 1, f);
        test_labels = (uint8_t*)malloc(n * sizeof(uint8_t));
        fread(test_labels, sizeof(uint8_t), n, f);
        fclose(f);
    }
    
    if (!train_labels || !test_labels) {
        fprintf(stderr, "Failed to load labels\n");
        free(train_features);
        free(test_features);
        if (train_labels) free(train_labels);
        if (test_labels) free(test_labels);
        return 1;
    }
    
    printf("\n=== Training Classifier on Extracted Features ===\n");
    printf("Train features: (%d, %d)\n", train_num_samples, train_feature_size);
    printf("Test features: (%d, %d)\n\n", test_num_samples, test_feature_size);
    
    // Create and train classifier
    LogisticRegression* classifier = create_classifier(train_feature_size, FC2_OUTPUT_SIZE);
    train_classifier(classifier, train_features, train_labels, 
                    train_num_samples, 20, 0.001f);
    
    // Evaluate
    printf("\n=== Evaluation ===\n");
    float train_acc = predict_and_evaluate(classifier, train_features, train_labels, train_num_samples);
    printf("Training Accuracy: %.4f (%.2f%%)\n", train_acc, train_acc * 100);
    
    float test_acc = predict_and_evaluate(classifier, test_features, test_labels, test_num_samples);
    printf("Test Accuracy: %.4f (%.2f%%)\n", test_acc, test_acc * 100);
    
    // Cleanup
    free_classifier(classifier);
    free(train_features);
    free(test_features);
    free(train_labels);
    free(test_labels);
    
    return 0;
}
