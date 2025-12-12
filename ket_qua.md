## CPU voi 1 data_batch:
```bash
Epoch 10/10:
  Progress: [========================================] 156/156 (100.0%) - Loss: 2.1918, Acc: 0.2489
  Epoch completed in 240.16 seconds - Avg Loss: 2.1918, Avg Acc: 0.2489
Evaluating on 10000 samples...
Test Loss: 2.1879, Test Accuracy: 0.2575

=== Training Complete ===
Total training time: 2659.90 seconds
Average time per epoch: 265.99 seconds
```

***
## GPU naive

```bash
=== CIFAR-10 CNN Training (GPU Naive Implementation) ===
Batch size: 64
Learning rate: 0.0010
Number of epochs: 20

Using GPU: Tesla T4
Compute Capability: 7.5
Global Memory: 14.74 GB

Loading training data...
Loading ../cifar-10-batches-bin/data_batch_1.bin...
Loading ../cifar-10-batches-bin/data_batch_2.bin...
Loading ../cifar-10-batches-bin/data_batch_3.bin...
Loading ../cifar-10-batches-bin/data_batch_4.bin...
Loading ../cifar-10-batches-bin/data_batch_5.bin...
Loaded 50000 training images
Loading test data...
Loading ../cifar-10-batches-bin/test_batch.bin...
Loaded 10000 test images

Creating CNN model on GPU...
Weights initialized and copied to GPU

Model Architecture:
  Input: 32x32x3
  Conv1: 32 filters, 3x3 kernel -> 32x32x32
  Pool1: 2x2 -> 16x16x32
  Conv2: 128 filters, 3x3 kernel -> 16x16x128
  Pool2: 2x2 -> 8x8x128
  FC1: 8192 -> 128
  FC2: 128 -> 10 (output)

Epoch 1/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.2904, Acc: 0.1254
  Epoch completed in 109.49 seconds - Avg Loss: 2.2904, Avg Acc: 0.1254
Evaluating on 10000 samples...
Test Loss: 2.2803, Test Accuracy: 0.1458

Epoch 2/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.2715, Acc: 0.1589
  Epoch completed in 110.33 seconds - Avg Loss: 2.2715, Avg Acc: 0.1589
Evaluating on 10000 samples...
Test Loss: 2.2607, Test Accuracy: 0.1824

Epoch 3/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.2497, Acc: 0.1937
  Epoch completed in 110.91 seconds - Avg Loss: 2.2497, Avg Acc: 0.1937
Evaluating on 10000 samples...
Test Loss: 2.2363, Test Accuracy: 0.1998

Epoch 4/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.2224, Acc: 0.2144
  Epoch completed in 110.96 seconds - Avg Loss: 2.2224, Avg Acc: 0.2144
Evaluating on 10000 samples...
Test Loss: 2.2056, Test Accuracy: 0.2135

Epoch 5/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.1896, Acc: 0.2295
  Epoch completed in 110.94 seconds - Avg Loss: 2.1896, Avg Acc: 0.2295
Evaluating on 10000 samples...
Test Loss: 2.1703, Test Accuracy: 0.2239

Epoch 6/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.1537, Acc: 0.2435
  Epoch completed in 110.95 seconds - Avg Loss: 2.1537, Avg Acc: 0.2435
Evaluating on 10000 samples...
Test Loss: 2.1336, Test Accuracy: 0.2359

Epoch 7/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.1186, Acc: 0.2549
  Epoch completed in 110.94 seconds - Avg Loss: 2.1186, Avg Acc: 0.2549
Evaluating on 10000 samples...
Test Loss: 2.0996, Test Accuracy: 0.2479

Epoch 8/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.0878, Acc: 0.2624
  Epoch completed in 110.94 seconds - Avg Loss: 2.0878, Avg Acc: 0.2624
Evaluating on 10000 samples...
Test Loss: 2.0718, Test Accuracy: 0.2557

Epoch 9/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.0638, Acc: 0.2675
  Epoch completed in 110.92 seconds - Avg Loss: 2.0638, Avg Acc: 0.2675
Evaluating on 10000 samples...
Test Loss: 2.0512, Test Accuracy: 0.2636

Epoch 10/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.0455, Acc: 0.2730
  Epoch completed in 110.91 seconds - Avg Loss: 2.0455, Avg Acc: 0.2730
Evaluating on 10000 samples...
Test Loss: 2.0354, Test Accuracy: 0.2694

Epoch 11/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.0307, Acc: 0.2779
  Epoch completed in 110.90 seconds - Avg Loss: 2.0307, Avg Acc: 0.2779
Evaluating on 10000 samples...
Test Loss: 2.0220, Test Accuracy: 0.2753

Epoch 12/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.0174, Acc: 0.2836
  Epoch completed in 110.93 seconds - Avg Loss: 2.0174, Avg Acc: 0.2836
Evaluating on 10000 samples...
Test Loss: 2.0093, Test Accuracy: 0.2810

Epoch 13/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.0044, Acc: 0.2888
  Epoch completed in 110.93 seconds - Avg Loss: 2.0044, Avg Acc: 0.2888
Evaluating on 10000 samples...
Test Loss: 1.9968, Test Accuracy: 0.2859

Epoch 14/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.9914, Acc: 0.2941
  Epoch completed in 110.92 seconds - Avg Loss: 1.9914, Avg Acc: 0.2941
Evaluating on 10000 samples...
Test Loss: 1.9843, Test Accuracy: 0.2917

Epoch 15/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.9783, Acc: 0.2983
  Epoch completed in 110.99 seconds - Avg Loss: 1.9783, Avg Acc: 0.2983
Evaluating on 10000 samples...
Test Loss: 1.9716, Test Accuracy: 0.2975

Epoch 16/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.9649, Acc: 0.3034
  Epoch completed in 110.95 seconds - Avg Loss: 1.9649, Avg Acc: 0.3034
Evaluating on 10000 samples...
Test Loss: 1.9588, Test Accuracy: 0.3009

Epoch 17/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.9512, Acc: 0.3085
  Epoch completed in 110.96 seconds - Avg Loss: 1.9512, Avg Acc: 0.3085
Evaluating on 10000 samples...
Test Loss: 1.9456, Test Accuracy: 0.3050

Epoch 18/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.9373, Acc: 0.3126
  Epoch completed in 110.94 seconds - Avg Loss: 1.9373, Avg Acc: 0.3126
Evaluating on 10000 samples...
Test Loss: 1.9321, Test Accuracy: 0.3103

Epoch 19/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.9232, Acc: 0.3182
  Epoch completed in 110.92 seconds - Avg Loss: 1.9232, Avg Acc: 0.3182
Evaluating on 10000 samples...
Test Loss: 1.9182, Test Accuracy: 0.3153

Epoch 20/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.9091, Acc: 0.3234
  Epoch completed in 110.92 seconds - Avg Loss: 1.9091, Avg Acc: 0.3234
Evaluating on 10000 samples...
Test Loss: 1.9044, Test Accuracy: 0.3205

=== Training Complete ===
Total training time: 2216.64 seconds
Average time per epoch: 110.83 seconds
```
***
## Extract and Test feature
```bash
=== Extracting Training Features ===
Extracting features from 50000 samples...
  Processed 782/782 batches (100.0%)
Feature extraction complete: 50000 samples x 8192 features
Saving features to train_features.bin...
Features saved: 50000 samples x 8192 features

=== Extracting Test Features ===
Extracting features from 10000 samples...
  Processed 157/157 batches (100.0%)
Feature extraction complete: 10000 samples x 8192 features
Saving features to test_features.bin...
Features saved: 10000 samples x 8192 features

=== Training Classifier on Extracted Features ===
Train features: (50000, 8192)
Test features: (10000, 8192)


Training classifier...
Samples: 50000, Features: 8192, Classes: 10
Epochs: 20, Learning rate: 0.0100

Epoch 1/20 - Loss: 5.2913 - Time: 9.62s
Epoch 2/20 - Loss: 5.0144 - Time: 9.73s
Epoch 3/20 - Loss: 4.9034 - Time: 9.72s
Epoch 4/20 - Loss: 4.8100 - Time: 9.26s
Epoch 5/20 - Loss: 4.7521 - Time: 9.80s
Epoch 6/20 - Loss: 4.7033 - Time: 9.63s
Epoch 7/20 - Loss: 4.6474 - Time: 9.60s
Epoch 8/20 - Loss: 4.6138 - Time: 9.15s
Epoch 9/20 - Loss: 4.5843 - Time: 9.65s
Epoch 10/20 - Loss: 4.5543 - Time: 9.58s
Epoch 11/20 - Loss: 4.5262 - Time: 9.78s
Epoch 12/20 - Loss: 4.5037 - Time: 9.63s
Epoch 13/20 - Loss: 4.4832 - Time: 9.34s
Epoch 14/20 - Loss: 4.4579 - Time: 9.73s
Epoch 15/20 - Loss: 4.4379 - Time: 9.77s
Epoch 16/20 - Loss: 4.4271 - Time: 9.65s
Epoch 17/20 - Loss: 4.4035 - Time: 9.19s
Epoch 18/20 - Loss: 4.3884 - Time: 9.65s
Epoch 19/20 - Loss: 4.3676 - Time: 9.69s
Epoch 20/20 - Loss: 4.3541 - Time: 9.64s

=== Evaluation ===
Training Accuracy: 0.2672 (26.72%)
Test Accuracy: 0.2624 (26.24%)
```