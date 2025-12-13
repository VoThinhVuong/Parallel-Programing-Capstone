# Step 3: Feature Extraction & Classification

This folder now includes **Step 3** implementation: Extract features using the trained encoder and train a classifier.

## Overview

After training the CNN, we can use the encoder part (Conv1→Pool1→Conv2→Pool2) to extract features from images. These features can then be used to train a simpler classifier.

### Architecture

**Encoder (Feature Extractor)**:
```
Input: 32×32×3
├─ Conv1: 32 filters, 3×3 → 32×32×32
├─ ReLU
├─ Pool1: 2×2 → 16×16×32
├─ Conv2: 128 filters, 3×3 → 16×16×128
├─ ReLU
└─ Pool2: 2×2 → 8×8×128 (Features: 8192-dimensional)
```

**Classifier**: Logistic Regression on extracted features (8192 → 10)

## Files Added

- `feature_extractor.cuh/cu` - Feature extraction functions
- `extract_features.cu` - Main program for feature extraction and classification
- Updated `main.cu` - Now saves encoder weights after training

## Workflow

### 1. Train the CNN (if not already done)

```bash
make clean
make
make run
```

This will:
- Train the full CNN for 10 epochs
- Save encoder weights to `encoder_weights.bin`

### 2. Extract Features and Train Classifier

```bash
make extract
```

This will:
- Load the trained encoder weights
- Extract features from training set (50,000 × 4096)
- Extract features from test set (10,000 × 4096)
- Save features to `train_features.bin` and `test_features.bin`
- Train a logistic regression classifier on the features
- Report test accuracy

## Feature Extraction Details

### Feature Size
- **Pool2 output**: 128 channels × 8×8 = **8192 features** per image
- **Training features**: (50,000, 8192)
- **Test features**: (10,000, 8192)

### Encoder Weights
The encoder includes:
- Conv1 weights and biases
- Conv2 weights and biases

These are saved after training and loaded for feature extraction.

### Saved Files

After running the programs:
- `encoder_weights.bin` - Trained encoder parameters
- `train_features.bin` - Extracted training features (50,000 × 8192)
- `test_features.bin` - Extracted test features (10,000 × 8192)
- `train_labels.bin` - Training labels
- `test_labels.bin` - Test labels

## Expected Results

### Full CNN Training (10 epochs)
- Training accuracy: ~25-35%
- Test accuracy: ~20-30%

### Feature-based Classifier (20 epochs)
- Using features from trained encoder
- Training accuracy: Similar to encoder
- Test accuracy: Should match or slightly improve CNN

## Usage Examples

### Complete Pipeline

```bash
# 1. Train CNN
make clean
make run

# Wait for training to complete...
# Encoder weights saved to encoder_weights.bin

# 2. Extract features and train classifier
make extract

# First run extracts features (takes a few minutes)
# Subsequent runs can reuse saved features
```

### Re-train Classifier Only

If you already have extracted features:

```bash
# Run extract_features
./extract_features

# When prompted:
# Re-extract features? (y/n): n
```

This will load existing features and re-train the classifier.

## Performance

### Feature Extraction Speed
- GPU: ~30-60 seconds for 50,000 images
- Depends on GPU and batch size

### Classifier Training Speed
- CPU: ~5-10 seconds for 20 epochs
- Simple logistic regression is fast

## Comparing Results

| Method | Architecture | Test Accuracy |
|--------|-------------|---------------|
| Full CNN | Conv→Pool→Conv→Pool→FC→FC | ~20-30% (10 epochs) |
| Feature Classifier | Encoder + Logistic Reg | ~20-30% (matches encoder) |

*Note: Low accuracy is expected with only 10 training epochs. Full training requires 50-100+ epochs.*

## Advantages of Feature-based Classification

1. **Faster Training**: Once features extracted, classifier trains quickly
2. **Less Memory**: Only 4096 features vs full 3072 pixels
3. **Transfer Learning**: Can reuse encoder for different tasks
4. **Experimentation**: Easy to try different classifiers

## Code Structure

### feature_extractor.cu

```cpp
// Save encoder weights after training
save_encoder_weights(cnn, "encoder_weights.bin");

// Load encoder weights
load_encoder_weights(cnn, "encoder_weights.bin");

// Extract features from dataset
float* features = extract_features(cnn, dataset, batch_size);
// Returns: num_samples × 8192 array

// Run encoder forward pass only
encoder_forward_pass(cnn, d_input);
// Output in: cnn->pool2->d_output (8192 features per sample)
```

### extract_features.cu

```cpp
// Simple logistic regression classifier
LogisticRegression* clf = create_classifier(feature_size, num_classes);

// Train on extracted features
train_classifier(clf, features, labels, num_samples, epochs, lr);

// Evaluate
float accuracy = predict_and_evaluate(clf, features, labels, num_samples);
```

## Troubleshooting

### "Cannot open encoder_weights.bin"
- Run the main training program first: `make run`
- Weights are saved automatically after training

### Out of memory during feature extraction
- Reduce batch size in `extract_features.cu`
- Default is 64, try 32 or 16

### Low classifier accuracy
- Ensure encoder was properly trained
- Try training full CNN for more epochs
- Increase classifier training epochs

### Feature files taking too much space
- Features are ~1.6GB for training set (8192 features × 50,000 samples × 4 bytes)
- Can be deleted and re-extracted if needed
- Or use compressed storage (not implemented)

## Next Steps

After Step 3:
- Try different classifiers (SVM, Random Forest)
- Fine-tune the full CNN for better features
- Implement data augmentation
- Use features for transfer learning

## Project Alignment

This implements **Step 3** from the project requirements:
- ✅ Load trained encoder weights
- ✅ Run encoder forward pass (no decoder/FC layers)
- ✅ Extract features: (50000, 8192) and (10000, 8192)
- ✅ Train classifier on features
- ✅ Report test accuracy
