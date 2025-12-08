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
Number of epochs: 10

Using GPU: NVIDIA GeForce RTX 3050 Laptop GPU
Compute Capability: 8.6
Global Memory: 4.00 GB

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

Epoch 1/10:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.2880, Acc: 0.1085
  Epoch completed in 121.74 seconds - Avg Loss: 2.2880, Avg Acc: 0.1085
Evaluating on 10000 samples...
Test Loss: 2.2774, Test Accuracy: 0.1243

Epoch 2/10:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.2691, Acc: 0.1422
  Epoch completed in 120.66 seconds - Avg Loss: 2.2691, Avg Acc: 0.1422
Evaluating on 10000 samples...
Test Loss: 2.2578, Test Accuracy: 0.1598

Epoch 3/10:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.2487, Acc: 0.1677
  Epoch completed in 120.79 seconds - Avg Loss: 2.2487, Avg Acc: 0.1677
Evaluating on 10000 samples...
Test Loss: 2.2352, Test Accuracy: 0.1867

Epoch 4/10:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.2249, Acc: 0.1907
  Epoch completed in 120.77 seconds - Avg Loss: 2.2249, Avg Acc: 0.1907
Evaluating on 10000 samples...
Test Loss: 2.2088, Test Accuracy: 0.2057

Epoch 5/10:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.1975, Acc: 0.2123
  Epoch completed in 120.83 seconds - Avg Loss: 2.1975, Avg Acc: 0.2123
Evaluating on 10000 samples...
Test Loss: 2.1788, Test Accuracy: 0.2253

Epoch 6/10:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.1671, Acc: 0.2344
  Epoch completed in 120.62 seconds - Avg Loss: 2.1671, Avg Acc: 0.2344
Evaluating on 10000 samples...
Test Loss: 2.1462, Test Accuracy: 0.2433

Epoch 7/10:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.1349, Acc: 0.2508
  Epoch completed in 120.65 seconds - Avg Loss: 2.1349, Avg Acc: 0.2508
Evaluating on 10000 samples...
Test Loss: 2.1131, Test Accuracy: 0.2520

Epoch 8/10:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.1037, Acc: 0.2599
  Epoch completed in 120.69 seconds - Avg Loss: 2.1037, Avg Acc: 0.2599
Evaluating on 10000 samples...
Test Loss: 2.0830, Test Accuracy: 0.2602

Epoch 9/10:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.0767, Acc: 0.2658
  Epoch completed in 120.82 seconds - Avg Loss: 2.0767, Avg Acc: 0.2658
Evaluating on 10000 samples...
Test Loss: 2.0590, Test Accuracy: 0.2649

Epoch 10/10:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.0558, Acc: 0.2710
  Epoch completed in 120.76 seconds - Avg Loss: 2.0558, Avg Acc: 0.2710
Evaluating on 10000 samples...
Test Loss: 2.0414, Test Accuracy: 0.2709

=== Training Complete ===
Total training time: 1208.32 seconds
Average time per epoch: 120.83 seconds
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

Epoch 1/20 - Loss: 3.6125 - Time: 11.75s
Epoch 2/20 - Loss: 3.4160 - Time: 11.14s
Epoch 3/20 - Loss: 3.3302 - Time: 11.25s
Epoch 4/20 - Loss: 3.2713 - Time: 11.39s
Epoch 5/20 - Loss: 3.2254 - Time: 11.56s
Epoch 6/20 - Loss: 3.1873 - Time: 11.34s
Epoch 7/20 - Loss: 3.1556 - Time: 11.18s
Epoch 8/20 - Loss: 3.1282 - Time: 11.19s
Epoch 9/20 - Loss: 3.1044 - Time: 11.25s
Epoch 10/20 - Loss: 3.0832 - Time: 11.16s
Epoch 11/20 - Loss: 3.0640 - Time: 11.18s
Epoch 12/20 - Loss: 3.0468 - Time: 11.48s
Epoch 13/20 - Loss: 3.0312 - Time: 11.19s
Epoch 14/20 - Loss: 3.0168 - Time: 11.18s
Epoch 15/20 - Loss: 3.0034 - Time: 11.15s
Epoch 16/20 - Loss: 2.9907 - Time: 11.17s
Epoch 17/20 - Loss: 2.9791 - Time: 11.27s
Epoch 18/20 - Loss: 2.9681 - Time: 11.28s
Epoch 19/20 - Loss: 2.9578 - Time: 11.28s
Epoch 20/20 - Loss: 2.9481 - Time: 11.23s

=== Evaluation ===
Training Accuracy: 0.3670 (36.70%)
Test Accuracy: 0.3591 (35.91%)
```