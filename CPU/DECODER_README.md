# Decoder Implementation for Image Reconstruction

## Overview

This implementation adds a decoder network to the CNN that reconstructs the original CIFAR-10 images from the learned features at the Pool2 layer. The decoder is trained only during the **last epoch** to minimize training time while still enabling visualization of what the network has learned.

## Architecture

### Encoder (Existing)
- Input: 32×32×3 RGB image
- Conv1: 32 filters, 3×3 kernel → 32×32×32
- Pool1: 2×2 max pooling → 16×16×32
- Conv2: 128 filters, 3×3 kernel → 16×16×128
- Pool2: 2×2 max pooling → **8×8×128** (bottleneck features)
- FC1: 8192 → 128
- FC2: 128 → 10 (classification)

### Decoder (New)
The decoder mirrors the encoder path to reconstruct images:

- Input: **8×8×128** (from Pool2)
- Upsample1: Nearest neighbor 2× → 16×16×128
- TransposeConv1: 3×3 kernel, 128→64 filters + ReLU → 16×16×64
- Upsample2: Nearest neighbor 2× → 32×32×64
- TransposeConv2: 3×3 kernel, 64→3 filters → **32×32×3** (reconstructed RGB)

## Implementation Details

### New Data Structures

#### TransposeConvLayer
```c
typedef struct {
    int input_channels;
    int output_channels;
    int kernel_size;
    int stride;
    int padding;
    int input_size;
    int output_size;
    float* weights;              // [out_ch × in_ch × k × k]
    float* bias;                 // [out_ch]
    float* output;               // Forward pass output
    float* weight_gradients;     // Backprop gradients
    float* bias_gradients;
    float* input_gradients;
} TransposeConvLayer;
```

#### UpsampleLayer
```c
typedef struct {
    int channels;
    int input_size;
    int scale_factor;
    int output_size;
    float* output;               // Forward pass output
    float* input_gradients;      // Backprop gradients
} UpsampleLayer;
```

#### Decoder
```c
typedef struct {
    int batch_size;
    UpsampleLayer* upsample1;    // 8×8 → 16×16
    TransposeConvLayer* tconv1;  // 128ch → 64ch
    UpsampleLayer* upsample2;    // 16×16 → 32×32
    TransposeConvLayer* tconv2;  // 64ch → 3ch
    float* tconv1_relu;          // Activation after tconv1
    float* tconv1_relu_grad;     // Gradient for ReLU
    float* reconstructed;        // Final reconstructed images
} Decoder;
```

### Training Process

1. **Epochs 1 to N-1**: Train only the encoder and classifier (normal training)
2. **Last Epoch (Epoch N)**:
   - Create and initialize decoder
   - For each batch:
     - Forward pass through encoder + classifier
     - Train classifier (backward pass + weight update)
     - Forward pass through decoder
     - Calculate reconstruction loss (MSE)
     - Backward pass through decoder
     - Update decoder weights
   - Save all reconstructed images to binary file

### Loss Functions

**Classification Loss** (all epochs):
```
L_class = -log(p_y) where y is true label
```

**Reconstruction Loss** (last epoch only):
```
L_recon = MSE(reconstructed, original) = (1/N) Σ(recon_i - orig_i)²
```

## Usage

### Building
```bash
cd CPU
make
```

### Training with Decoder
```bash
./cnn 20  # Train for 20 epochs, decoder active in epoch 20
```

The decoder will be created automatically in the last epoch, and reconstructed images will be saved to:
```
extracted_features/reconstructed_images_cpu.bin
```

### Visualizing Reconstructions

Use the provided Python script to visualize and analyze reconstructions:

```bash
python visualize_reconstructions.py
```

This script provides:
1. **Comparison Grid**: Side-by-side view of original, reconstructed, and difference images
2. **Quality Distribution**: Histogram of PSNR and MSE across all test images
3. **Per-Class Quality**: Mean PSNR for each CIFAR-10 class
4. **Interactive Viewer**: Browse through all reconstructions with arrow keys

### Output Files

- `reconstructed_images_cpu.bin`: Binary file with all reconstructed images
- `reconstruction_comparison.png`: Grid showing sample reconstructions
- `reconstruction_quality_dist.png`: PSNR and MSE distributions
- `reconstruction_per_class.png`: Quality metrics per class

## File Format

### reconstructed_images_cpu.bin
```
[int32] num_images
[int32] image_size (3072 = 3×32×32)
[float32 × num_images × 3072] image data in CHW format (channels, height, width)
```

## Key Functions

### cnn.c
- `create_transpose_conv_layer()`: Create transpose convolution layer
- `create_upsample_layer()`: Create upsampling layer
- `create_decoder()`: Assemble complete decoder
- `initialize_decoder_weights()`: He initialization for decoder weights

### forward.c
- `upsample_forward()`: Nearest neighbor upsampling
- `transpose_conv_forward()`: Transpose convolution (deconvolution)
- `decoder_forward()`: Complete decoder forward pass

### backward.c
- `upsample_backward()`: Gradient redistribution for upsampling
- `transpose_conv_backward()`: Gradient computation for transpose conv
- `decoder_backward()`: Complete decoder backward pass
- `update_decoder_weights()`: SGD weight update for decoder

### main.c
- `calculate_reconstruction_loss()`: MSE between original and reconstructed
- `save_reconstructed_images()`: Write reconstructions to binary file
- `train_epoch()`: Modified to include decoder training in last epoch

## Performance Considerations

### Memory Usage
The decoder adds approximately:
- TransposeConv1 weights: 64×128×3×3 = 73,728 floats
- TransposeConv2 weights: 3×64×3×3 = 1,728 floats
- Intermediate activations: ~400KB per batch
- **Total overhead**: ~300KB weights + 400KB activations per batch

### Training Time
- Training only in last epoch minimizes overhead
- Decoder forward/backward adds ~30-40% time to last epoch
- Overall impact: <5% increase in total training time for 20+ epochs

## Metrics

### Expected Quality
Typical reconstruction quality on CIFAR-10:
- **PSNR**: 15-25 dB (higher is better)
- **MSE**: 0.001-0.01 (lower is better)

Note: These are approximate values. The decoder is trained for only one epoch, so reconstructions may not be perfect but should capture the main structure and colors.

### Interpretation
- **PSNR > 20 dB**: Good reconstruction quality
- **PSNR 15-20 dB**: Moderate quality, main features visible
- **PSNR < 15 dB**: Poor quality, but may still show structure

## Troubleshooting

### Decoder Not Created
Ensure you're running enough epochs. The decoder is only created in the last epoch.

### File Not Found
Check that the path `extracted_features/` exists. The program creates it automatically, but if you're running from a different directory, adjust the path in main.c.

### Poor Reconstruction Quality
This is expected with only one epoch of decoder training. To improve:
1. Increase number of epochs (decoder still trains only in last epoch)
2. Modify code to train decoder for multiple epochs
3. Adjust learning rate for decoder specifically

### Memory Issues
If you encounter memory errors:
1. Reduce batch size in main.c
2. Check available RAM
3. Verify all malloc calls succeed

## Future Improvements

1. **Multiple Epoch Training**: Train decoder for last N epochs instead of just one
2. **Separate Learning Rates**: Use different learning rates for encoder and decoder
3. **Better Upsampling**: Replace nearest neighbor with bilinear interpolation
4. **Skip Connections**: Add skip connections from encoder to decoder (U-Net style)
5. **Perceptual Loss**: Use feature-based loss instead of just MSE
6. **Progressive Training**: Gradually introduce decoder training over last few epochs

## References

- He et al. (2015): "Delving Deep into Rectifiers" - He initialization
- Dumoulin & Visin (2016): "A guide to convolution arithmetic for deep learning" - Transpose convolution
- Ronneberger et al. (2015): "U-Net: Convolutional Networks for Biomedical Image Segmentation" - Skip connections (future work)
