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
## GPU with shared memory

```bash
=== CIFAR-10 CNN Training (GPU Shared Memory Implementation) ===
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
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.2905, Acc: 0.1354
  Epoch completed in 109.38 seconds - Avg Loss: 2.2905, Avg Acc: 0.1354
Evaluating on 10000 samples...
Test Loss: 2.2805, Test Accuracy: 0.1565

Epoch 2/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.2719, Acc: 0.1647
  Epoch completed in 109.59 seconds - Avg Loss: 2.2719, Avg Acc: 0.1647
Evaluating on 10000 samples...
Test Loss: 2.2615, Test Accuracy: 0.1840

Epoch 3/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.2515, Acc: 0.1882
  Epoch completed in 110.18 seconds - Avg Loss: 2.2515, Avg Acc: 0.1882
Evaluating on 10000 samples...
Test Loss: 2.2386, Test Accuracy: 0.2015

Epoch 4/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.2265, Acc: 0.2028
  Epoch completed in 110.24 seconds - Avg Loss: 2.2265, Avg Acc: 0.2028
Evaluating on 10000 samples...
Test Loss: 2.2113, Test Accuracy: 0.2102

Epoch 5/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.1986, Acc: 0.2145
  Epoch completed in 109.75 seconds - Avg Loss: 2.1986, Avg Acc: 0.2145
Evaluating on 10000 samples...
Test Loss: 2.1817, Test Accuracy: 0.2167

Epoch 6/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.1684, Acc: 0.2270
  Epoch completed in 109.75 seconds - Avg Loss: 2.1684, Avg Acc: 0.2270
Evaluating on 10000 samples...
Test Loss: 2.1492, Test Accuracy: 0.2281

Epoch 7/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.1347, Acc: 0.2429
  Epoch completed in 109.75 seconds - Avg Loss: 2.1347, Avg Acc: 0.2429
Evaluating on 10000 samples...
Test Loss: 2.1137, Test Accuracy: 0.2441

Epoch 8/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.0997, Acc: 0.2570
  Epoch completed in 109.75 seconds - Avg Loss: 2.0997, Avg Acc: 0.2570
Evaluating on 10000 samples...
Test Loss: 2.0794, Test Accuracy: 0.2533

Epoch 9/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.0688, Acc: 0.2670
  Epoch completed in 109.75 seconds - Avg Loss: 2.0688, Avg Acc: 0.2670
Evaluating on 10000 samples...
Test Loss: 2.0521, Test Accuracy: 0.2627

Epoch 10/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.0449, Acc: 0.2750
  Epoch completed in 109.75 seconds - Avg Loss: 2.0449, Avg Acc: 0.2750
Evaluating on 10000 samples...
Test Loss: 2.0321, Test Accuracy: 0.2706

Epoch 11/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.0270, Acc: 0.2832
  Epoch completed in 109.75 seconds - Avg Loss: 2.0270, Avg Acc: 0.2832
Evaluating on 10000 samples...
Test Loss: 2.0167, Test Accuracy: 0.2764

Epoch 12/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.0122, Acc: 0.2892
  Epoch completed in 109.75 seconds - Avg Loss: 2.0122, Avg Acc: 0.2892
Evaluating on 10000 samples...
Test Loss: 2.0034, Test Accuracy: 0.2834

Epoch 13/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.9988, Acc: 0.2949
  Epoch completed in 109.75 seconds - Avg Loss: 1.9988, Avg Acc: 0.2949
Evaluating on 10000 samples...
Test Loss: 1.9909, Test Accuracy: 0.2884

Epoch 14/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.9858, Acc: 0.2993
  Epoch completed in 109.74 seconds - Avg Loss: 1.9858, Avg Acc: 0.2993
Evaluating on 10000 samples...
Test Loss: 1.9787, Test Accuracy: 0.2950

Epoch 15/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.9728, Acc: 0.3044
  Epoch completed in 109.75 seconds - Avg Loss: 1.9728, Avg Acc: 0.3044
Evaluating on 10000 samples...
Test Loss: 1.9664, Test Accuracy: 0.3001

Epoch 16/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.9598, Acc: 0.3085
  Epoch completed in 109.75 seconds - Avg Loss: 1.9598, Avg Acc: 0.3085
Evaluating on 10000 samples...
Test Loss: 1.9540, Test Accuracy: 0.3045

Epoch 17/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.9467, Acc: 0.3137
  Epoch completed in 109.75 seconds - Avg Loss: 1.9467, Avg Acc: 0.3137
Evaluating on 10000 samples...
Test Loss: 1.9416, Test Accuracy: 0.3078

Epoch 18/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.9334, Acc: 0.3182
  Epoch completed in 109.75 seconds - Avg Loss: 1.9334, Avg Acc: 0.3182
Evaluating on 10000 samples...
Test Loss: 1.9290, Test Accuracy: 0.3133

Epoch 19/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.9202, Acc: 0.3226
  Epoch completed in 109.75 seconds - Avg Loss: 1.9202, Avg Acc: 0.3226
Evaluating on 10000 samples...
Test Loss: 1.9164, Test Accuracy: 0.3179

Epoch 20/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.9072, Acc: 0.3277
  Epoch completed in 109.75 seconds - Avg Loss: 1.9072, Avg Acc: 0.3277
Evaluating on 10000 samples...
Test Loss: 1.9041, Test Accuracy: 0.3233

=== Training Complete ===
Total training time: 2195.36 seconds
Average time per epoch: 109.77 seconds
```

***
## GPU with optimized gradients

```bash
=== CIFAR-10 CNN Training (GPU v3 - Optimized Gradients) ===
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
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.2947, Acc: 0.1308
  Epoch completed in 18.32 seconds - Avg Loss: 2.2947, Avg Acc: 0.1308
Evaluating on 10000 samples...
Test Loss: 2.2876, Test Accuracy: 0.1313

Epoch 2/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.2814, Acc: 0.1462
  Epoch completed in 18.71 seconds - Avg Loss: 2.2814, Avg Acc: 0.1462
Evaluating on 10000 samples...
Test Loss: 2.2744, Test Accuracy: 0.1503

Epoch 3/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.2670, Acc: 0.1667
  Epoch completed in 19.23 seconds - Avg Loss: 2.2670, Avg Acc: 0.1667
Evaluating on 10000 samples...
Test Loss: 2.2587, Test Accuracy: 0.1702

Epoch 4/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.2495, Acc: 0.1859
  Epoch completed in 19.30 seconds - Avg Loss: 2.2495, Avg Acc: 0.1859
Evaluating on 10000 samples...
Test Loss: 2.2394, Test Accuracy: 0.1926

Epoch 5/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.2283, Acc: 0.2037
  Epoch completed in 18.99 seconds - Avg Loss: 2.2283, Avg Acc: 0.2037
Evaluating on 10000 samples...
Test Loss: 2.2162, Test Accuracy: 0.2107

Epoch 6/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.2037, Acc: 0.2208
  Epoch completed in 18.92 seconds - Avg Loss: 2.2037, Avg Acc: 0.2208
Evaluating on 10000 samples...
Test Loss: 2.1901, Test Accuracy: 0.2286

Epoch 7/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.1765, Acc: 0.2362
  Epoch completed in 19.02 seconds - Avg Loss: 2.1765, Avg Acc: 0.2362
Evaluating on 10000 samples...
Test Loss: 2.1609, Test Accuracy: 0.2441

Epoch 8/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.1461, Acc: 0.2507
  Epoch completed in 19.12 seconds - Avg Loss: 2.1461, Avg Acc: 0.2507
Evaluating on 10000 samples...
Test Loss: 2.1282, Test Accuracy: 0.2491

Epoch 9/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.1132, Acc: 0.2618
  Epoch completed in 19.09 seconds - Avg Loss: 2.1132, Avg Acc: 0.2618
Evaluating on 10000 samples...
Test Loss: 2.0949, Test Accuracy: 0.2599

Epoch 10/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.0819, Acc: 0.2705
  Epoch completed in 19.01 seconds - Avg Loss: 2.0819, Avg Acc: 0.2705
Evaluating on 10000 samples...
Test Loss: 2.0657, Test Accuracy: 0.2696

Epoch 11/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.0562, Acc: 0.2769
  Epoch completed in 19.01 seconds - Avg Loss: 2.0562, Avg Acc: 0.2769
Evaluating on 10000 samples...
Test Loss: 2.0435, Test Accuracy: 0.2734

Epoch 12/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.0366, Acc: 0.2825
  Epoch completed in 19.04 seconds - Avg Loss: 2.0366, Avg Acc: 0.2825
Evaluating on 10000 samples...
Test Loss: 2.0266, Test Accuracy: 0.2800

Epoch 13/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.0207, Acc: 0.2873
  Epoch completed in 19.04 seconds - Avg Loss: 2.0207, Avg Acc: 0.2873
Evaluating on 10000 samples...
Test Loss: 2.0122, Test Accuracy: 0.2850

Epoch 14/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.0063, Acc: 0.2914
  Epoch completed in 19.07 seconds - Avg Loss: 2.0063, Avg Acc: 0.2914
Evaluating on 10000 samples...
Test Loss: 1.9987, Test Accuracy: 0.2920

Epoch 15/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.9924, Acc: 0.2974
  Epoch completed in 19.06 seconds - Avg Loss: 1.9924, Avg Acc: 0.2974
Evaluating on 10000 samples...
Test Loss: 1.9854, Test Accuracy: 0.2973

Epoch 16/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.9785, Acc: 0.3025
  Epoch completed in 19.06 seconds - Avg Loss: 1.9785, Avg Acc: 0.3025
Evaluating on 10000 samples...
Test Loss: 1.9721, Test Accuracy: 0.3038

Epoch 17/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.9645, Acc: 0.3080
  Epoch completed in 19.06 seconds - Avg Loss: 1.9645, Avg Acc: 0.3080
Evaluating on 10000 samples...
Test Loss: 1.9586, Test Accuracy: 0.3100

Epoch 18/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.9503, Acc: 0.3116
  Epoch completed in 19.06 seconds - Avg Loss: 1.9503, Avg Acc: 0.3116
Evaluating on 10000 samples...
Test Loss: 1.9450, Test Accuracy: 0.3124

Epoch 19/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.9360, Acc: 0.3167
  Epoch completed in 19.06 seconds - Avg Loss: 1.9360, Avg Acc: 0.3167
Evaluating on 10000 samples...
Test Loss: 1.9314, Test Accuracy: 0.3172

Epoch 20/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.9219, Acc: 0.3211
  Epoch completed in 19.05 seconds - Avg Loss: 1.9219, Avg Acc: 0.3211
Evaluating on 10000 samples...
Test Loss: 1.9180, Test Accuracy: 0.3195

=== Training Complete ===
Total training time: 380.23 seconds
Average time per epoch: 19.01 seconds
```

***
## Extract and Test feature
```bash
=== CIFAR-10 Feature Extraction & Classification ===

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

Creating CNN model...
Loading encoder weights from encoder_weights.bin...
Encoder weights loaded successfully

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
Epochs: 20, Learning rate: 0.0010

Epoch 1/20 - Loss: 2.0725 - Time: 9.17s
Epoch 2/20 - Loss: 1.9618 - Time: 9.19s
Epoch 3/20 - Loss: 1.9151 - Time: 9.10s
Epoch 4/20 - Loss: 1.8835 - Time: 8.75s
Epoch 5/20 - Loss: 1.8595 - Time: 9.18s
Epoch 6/20 - Loss: 1.8403 - Time: 9.29s
Epoch 7/20 - Loss: 1.8242 - Time: 8.92s
Epoch 8/20 - Loss: 1.8104 - Time: 8.84s
Epoch 9/20 - Loss: 1.7984 - Time: 9.17s
Epoch 10/20 - Loss: 1.7877 - Time: 9.22s
Epoch 11/20 - Loss: 1.7780 - Time: 8.86s
Epoch 12/20 - Loss: 1.7693 - Time: 9.06s
Epoch 13/20 - Loss: 1.7612 - Time: 9.19s
Epoch 14/20 - Loss: 1.7538 - Time: 9.32s
Epoch 15/20 - Loss: 1.7469 - Time: 8.87s
Epoch 16/20 - Loss: 1.7405 - Time: 9.13s
Epoch 17/20 - Loss: 1.7345 - Time: 9.21s
Epoch 18/20 - Loss: 1.7288 - Time: 9.15s
Epoch 19/20 - Loss: 1.7234 - Time: 8.85s
Epoch 20/20 - Loss: 1.7183 - Time: 9.12s

=== Evaluation ===
Training Accuracy: 0.3422 (34.22%)
Test Accuracy: 0.3367 (33.67%)
```
***
## SVM using cuML

* Evaluation
```bash
============================================================
SVM Evaluation with cuML (GPU-accelerated)
============================================================
Configuration:
  Test features: ../extracted_features/test_features_v3.bin
  Test labels: ../extracted_features/test_labels.bin
  Model: ./models/svm_model.pkl

Loading features from ../extracted_features/test_features_v3.bin...
  Number of samples: 10000
  Feature dimension: 8192
Loading labels from ../extracted_features/test_labels.bin...
  Number of labels: 10000
Loaded features shape: (10000, 8192)
Loaded labels shape: (10000,)
Loading cuML model from ./models/svm_model.pkl...
✓ Model loaded successfully!

============================================================
Evaluating SVM on Test Data (GPU-accelerated)
============================================================
Test samples: 10000
Test features: 8192

Transferring test data to GPU...
Data transfer to GPU: 7.75 seconds

Predicting on GPU...
✓ Prediction completed in 17.67 seconds
  Average time per sample: 1.77 ms

============================================================
Evaluation Results
============================================================
Test Accuracy: 40.91%

Per-Class Accuracy:
  airplane     (Class 0): 52.90%
  automobile   (Class 1): 43.00%
  bird         (Class 2): 29.10%
  cat          (Class 3): 21.40%
  deer         (Class 4): 27.50%
  dog          (Class 5): 27.80%
  frog         (Class 6): 52.40%
  horse        (Class 7): 52.60%
  ship         (Class 8): 56.80%
  truck        (Class 9): 45.60%

Detailed Classification Report:
              precision    recall  f1-score   support

    airplane     0.4401    0.5290    0.4805      1000
  automobile     0.5599    0.4300    0.4864      1000
        bird     0.2881    0.2910    0.2896      1000
         cat     0.3204    0.2140    0.2566      1000
        deer     0.3895    0.2750    0.3224      1000
         dog     0.3484    0.2780    0.3092      1000
        frog     0.3850    0.5240    0.4439      1000
       horse     0.3823    0.5260    0.4428      1000
        ship     0.5040    0.5680    0.5341      1000
       truck     0.4634    0.4560    0.4597      1000

    accuracy                         0.4091     10000
   macro avg     0.4081    0.4091    0.4025     10000
weighted avg     0.4081    0.4091    0.4025     10000


Confusion Matrix (10x10):
[[529  18  63  22  22  24  35  40 192  55]
 [ 87 430  17  33  24  21  43  36  79 230]
 [115  18 291  47 103  79 159 136  35  17]
 [ 64  31 121 214  57 173 148 110  41  41]
 [ 34  11 130  44 275  66 206 201  22  11]
 [ 39  17 138 151  41 278 124 162  31  19]
 [ 29  21 119  77  73  52 524  65  14  26]
 [ 45  21  76  43  84  66  59 526  23  57]
 [164  52  30  19  15  14  37  29 568  72]
 [ 96 149  25  18  12  25  26  71 122 456]]
```