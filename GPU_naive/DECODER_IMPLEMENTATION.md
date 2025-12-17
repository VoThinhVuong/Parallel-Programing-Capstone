# GPU Decoder Implementation

## Overview
This document describes the decoder implementation for the GPU_naive version of the CIFAR-10 CNN, matching the CPU implementation with joint training capabilities.

## Key Features

### 1. Joint Training
- Both classification loss and reconstruction loss update encoder weights
- Gradients from both losses are accumulated before updating encoder weights
- Decoder trains in **all epochs**, not just the last one
- Reconstructed images are saved only in the last epoch

### 2. Architecture

#### Encoder (Conv layers)
```
Input (32×32×3) → Conv1 (32 filters) → Pool1 → Conv2 (128 filters) → Pool2 (8×8×128)
```

#### Classifier (FC layers)
```
Pool2 → FC1 (128 units) → FC2 (10 units) → Softmax
```

#### Decoder
```
Pool2 (8×8×128) → Upsample1 (16×16) → TransConv1 (16×16×64) → 
Upsample2 (32×32) → TransConv2 (32×32×3)
```

### 3. Training Flow
For each batch:
1. **Forward pass** through encoder + classifier
2. Calculate classification loss and accuracy
3. **Backward pass** for classification
4. **Update classifier weights** (FC1, FC2 only)
5. **Decoder forward pass** (from Pool2 output)
6. Calculate reconstruction loss (MSE)
7. Compute reconstruction gradient on device
8. **Decoder backward pass**
9. **Backpropagate reconstruction gradients** through encoder
10. **Update decoder weights**
11. **Update encoder weights** (with accumulated gradients from both losses)

### 4. Loss Functions
- **Classification Loss**: Cross-entropy
- **Reconstruction Loss**: Mean Squared Error (MSE)
- Both losses are displayed in the progress bar

## File Structure

### New Files
- **decoder.cu**: All decoder-specific CUDA kernels and host functions
  - Upsample forward/backward kernels
  - Transpose convolution forward/backward kernels
  - Host wrapper functions

### Modified Files
- **cnn.cuh**: Added TransposeConvLayer, UpsampleLayer, Decoder structures
- **cnn.cu**: Added decoder creation/cleanup functions
- **forward.cuh/cu**: Added decoder forward pass declarations
- **backward.cuh/cu**: Added decoder backward pass and split weight update functions
- **main.cu**: Integrated decoder training into main training loop
- **Makefile**: Added decoder.cu to build

## Key Functions

### Decoder Creation
```c
Decoder* create_decoder(int batch_size);
void initialize_decoder_weights(Decoder* decoder);
void free_decoder(Decoder* decoder);
```

### Forward Pass
```c
void decoder_forward(Decoder* decoder, float* d_input, int batch_size);
```

### Backward Pass
```c
void decoder_backward(Decoder* decoder, float* d_output_gradient, float* d_input, int batch_size);
void backprop_reconstruction_to_encoder(CNN* cnn, float* d_input, float* d_pool2_gradient, int batch_size);
```

### Weight Updates
```c
void update_classifier_weights(CNN* cnn, float learning_rate);  // FC layers only
void update_encoder_weights(CNN* cnn, float learning_rate);     // Conv layers only
void update_decoder_weights(Decoder* decoder, float learning_rate);
```

## CUDA Kernels

### Upsample (Nearest Neighbor)
- **Forward**: Each output pixel copies value from corresponding input pixel
- **Backward**: Sum gradients from all output pixels that mapped to each input pixel

### Transpose Convolution
- **Forward**: Similar to standard convolution but with reversed input/output roles
- **Backward**: Uses atomicAdd for gradient accumulation (thread-safe)

### Gradient Accumulation
- **accumulate_gradients_kernel**: Adds gradients from reconstruction loss to existing classification gradients

## Reconstruction Loss Calculation

1. **Decoder forward** produces reconstructed images on device
2. **Copy to host** for loss calculation
3. **Calculate MSE** on host: `sum((original - reconstructed)^2) / size`
4. **Compute gradient** on device: `2 * (reconstructed - original)`
5. **Backward through decoder** using this gradient

## Output

### Progress Bar
Shows both classification and reconstruction losses:
```
Progress: [=========>...] 50/100 (50.0%) - Class Loss: 1.2345, Acc: 0.6789, Recon Loss: 0.012345
```

### Reconstructed Images
In the last epoch, first 10 batches save reconstructed images as PPM files:
```
reconstructed_images/reconstructed_batch0_sample0.ppm
reconstructed_images/reconstructed_batch0_sample1.ppm
...
```

## Feature Extraction Preservation
- **extract_features.cu** and **feature_extractor.cu** remain unchanged
- Separate build target for feature extraction
- Can still extract features and train SVM separately

## Compilation
```bash
make clean
make
```

This builds:
- `cifar10_cnn_gpu.exe`: Main training program with decoder
- `extract_features.exe`: Feature extraction tool (unchanged)

## Usage
```bash
# Train with decoder (default 20 epochs)
./cifar10_cnn_gpu.exe

# Train with custom number of epochs
./cifar10_cnn_gpu.exe 50
```

## Notes

1. **Memory Usage**: Decoder adds significant memory overhead (~4 additional layers + activation buffers)
2. **Performance**: Joint training is slower due to decoder forward/backward passes
3. **Gradient Accumulation**: Critical for joint training - encoder sees gradients from both losses
4. **Thread Safety**: atomicAdd used in transpose convolution backward pass to handle concurrent gradient writes
