# Extracted Features Directory

This directory contains the CNN-extracted features from both GPU implementations for SVM training.

## Directory Structure

```
extracted_features/
├── train_features_naive.bin    # Training features from GPU_naive (50,000 samples × 128 features)
├── train_features_shared.bin   # Training features from GPU_v2_shared_mem (50,000 samples × 128 features)
├── train_features_v3.bin       # Training features from GPU_v3_optimized_gradient (50,000 samples × 128 features)
├── test_features_naive.bin     # Test features from GPU_naive (10,000 samples × 128 features)
├── test_features_shared.bin    # Test features from GPU_v2_shared_mem (10,000 samples × 128 features)
├── test_features_v3.bin        # Test features from GPU_v3_optimized_gradient (10,000 samples × 128 features)
├── train_labels.bin            # Training labels (50,000 samples) - shared by all versions
└── test_labels.bin             # Test labels (10,000 samples) - shared by all versions
```

## File Format

### Feature Files
- **Header**: 4 bytes (int32) - number of samples
- **Data**: float32 array of shape (num_samples, 128)
- Features are extracted from the first fully connected layer (FC1) of the CNN

### Label Files
- **Header**: 4 bytes (int32) - number of samples
- **Data**: uint8 array of shape (num_samples,)
- Labels range from 0-9 representing CIFAR-10 classes

## Generating Features

### From GPU_naive
```bash
cd ../GPU_naive
make extract
./extract_features_naive.exe
```

### From GPU_v2_shared_mem
```bash
cd ../GPU_v2_shared_mem
make extract
./extract_features_shared.exe
```

### From GPU_v3_optimized_gradient
```bash
cd ../GPU_v3_optimized_gradient
make extract
./extract_features_v3.exe
```

## Using with SVM

### Train SVM on GPU_naive features
```bash
cd ../SVM
python train_svm.py naive
```
Train SVM on GPU_v3_optimized_gradient features
```bash
cd ../SVM
python train_svm.py v3
```

### Evaluate
```bash
python evaluate_svm.py naive   # For GPU_naive features
python evaluate_svm.py shared  # For GPU_v2_shared_mem features
python evaluate_svm.py v3      # For GPU_v3_optimized_gradient
python train_svm.py shared
```

### Evaluate
``All GPU versions produce the same labels since they use the same dataset
- Feature vectors may differ slightly due to different implementations and optimizations
- The features represent learned representations from the trained CNN encoder
- Expected feature dimension: 128 (from FC1 layer output)
- GPU_v3_optimized_gradient uses atomic-free gradient computation for improved performance

## Notes

- Both GPU versions produce the same labels since they use the same dataset
- Feature vectors may differ slightly due to different implementations and optimizations
- The features represent learned representations from the trained CNN encoder
- Expected feature dimension: 128 (from FC1 layer output)
