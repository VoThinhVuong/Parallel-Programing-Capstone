# Joint Training Implementation

## Overview

The decoder now implements **joint training** where the reconstruction loss backpropagates through the encoder, allowing both classification and reconstruction losses to update encoder weights.

## What Changed

### Before (Separate Training)
```
1. Classification: Input → Encoder → Classifier → Loss
   Gradients update: Encoder + Classifier

2. Reconstruction: Encoder output → Decoder → Loss
   Gradients update: Decoder only
```

### After (Joint Training)
```
1. Classification: Input → Encoder → Classifier → Loss
   Gradients accumulate in: Encoder + update Classifier

2. Reconstruction: Encoder output → Decoder → Loss
   Gradients accumulate in: Encoder + update Decoder

3. Final: Update Encoder with combined gradients
```

## Implementation Details

### Training Flow ([main.c](main.c))

```c
// 1. Forward classification
forward_pass(cnn, batch_images);

// 2. Backward classification (accumulates encoder gradients)
backward_pass(cnn, batch_images, batch_labels);

// 3. Update classifier weights immediately
update_classifier_weights(cnn, learning_rate);

// 4. Forward reconstruction
decoder_forward(cnn->decoder, cnn->pool2->output);

// 5. Backward reconstruction (accumulates MORE encoder gradients)
decoder_backward(cnn->decoder, cnn->pool2->output, recon_grad);
backprop_reconstruction_to_encoder(cnn, batch_images, 
                                   cnn->decoder->upsample1->input_gradients, 
                                   batch_size);

// 6. Update decoder weights
update_decoder_weights(cnn->decoder, learning_rate);

// 7. Update encoder weights with COMBINED gradients
update_encoder_weights(cnn, learning_rate);
```

### Key Functions ([backward.c](backward.c))

#### `backprop_reconstruction_to_encoder()`
New function that propagates reconstruction gradients backward through encoder layers:

```c
Pool2 gradient → Pool2 backward → Conv2 ReLU (accumulate) → 
Conv2 backward → Pool1 backward → Conv1 ReLU (accumulate) → Conv1 backward
```

**Important**: Uses `+=` to **accumulate** gradients, not replace them!

#### `update_classifier_weights()`
Updates only FC1 and FC2 layers (classifier head).

#### `update_encoder_weights()`
Updates only Conv1 and Conv2 layers (encoder) with accumulated gradients from:
- Classification loss gradients
- Reconstruction loss gradients

### Gradient Accumulation

For each encoder layer, gradients are accumulated:

```c
// From classification backward pass
cnn->conv2_relu_grad[i] = gradient_from_classification;

// From reconstruction backward pass (ACCUMULATE)
if (cnn->conv2_relu[i] > 0) {
    cnn->conv2_relu_grad[i] += gradient_from_reconstruction;
}

// Final weight update uses combined gradient
cnn->conv2->weights[i] -= learning_rate * cnn->conv2->weight_gradients[i];
```

## Benefits of Joint Training

### 1. **Better Feature Learning**
- Encoder learns features good for BOTH tasks
- Features preserve information while being discriminative

### 2. **Regularization Effect**
- Reconstruction constraint prevents overfitting
- Forces encoder to retain useful information

### 3. **Complementary Objectives**
- **Classification**: Learn what's different between classes
- **Reconstruction**: Learn what's preserved in the data
- **Combined**: Balanced feature representations

## Mathematical Formulation

### Encoder Gradient

```
∂L_total/∂θ_encoder = ∂L_classification/∂θ_encoder + ∂L_reconstruction/∂θ_encoder

Where:
- L_classification = CrossEntropy(y_pred, y_true)
- L_reconstruction = MSE(x_input, x_reconstructed)
- θ_encoder = weights of Conv1 and Conv2
```

### Weight Update

```
θ_encoder(t+1) = θ_encoder(t) - η * (∇L_class + ∇L_recon)

Where:
- η = learning rate
- ∇L_class = gradient from classification loss
- ∇L_recon = gradient from reconstruction loss
```

## Training Output

```
Epoch 1/10:
  Progress: [========] 156/156 (100.0%) - Class Loss: 2.1234, Acc: 0.2489, Recon Loss: 0.012345
  Epoch completed in 240.16 seconds - Class Loss: 2.1234, Acc: 0.2489, Recon Loss: 0.012345

Both losses now affect encoder training!
```

## Verification

To verify joint training is working:

1. **Encoder receives 2 gradient sources**: Check that `conv_backward()` is called twice per batch
2. **Gradients accumulate**: Check that `+=` is used for ReLU gradients
3. **Encoder updates once**: `update_encoder_weights()` called once per batch
4. **Reconstruction loss decreases**: Encoder learns to produce features good for reconstruction

## Files Modified

- **[main.c](main.c)**: Split weight updates, add reconstruction backprop call
- **[backward.c](backward.c)**: Implement `backprop_reconstruction_to_encoder()`, split weight update functions
- **[backward.h](backward.h)**: Add new function declarations

## Comparison with Previous Version

| Aspect | Separate Training | Joint Training |
|--------|------------------|----------------|
| Encoder gradients | Classification only | Classification + Reconstruction |
| Feature quality | Discriminative only | Discriminative + Information-preserving |
| Overfitting risk | Higher | Lower (regularization) |
| Computational cost | Lower | Slightly higher |
| PDF requirement | ❌ Incorrect | ✅ Correct |

## Expected Improvements

With joint training, expect:
- Slightly better test accuracy (regularization effect)
- Better reconstruction quality (encoder learns to preserve info)
- More robust features (less overfitting)
- Better generalization to test set
