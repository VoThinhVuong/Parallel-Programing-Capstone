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


***
# WITH LEARNING RATE = 0.01

## GPU v3
```bash
=== CIFAR-10 CNN Training (GPU v3 - Optimized Gradients) ===
Batch size: 64
Learning rate: 0.0100
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
  Progress: [========================================] 781/781 (100.0%) - Loss: 2.2012, Acc: 0.1996
  Epoch completed in 17.52 seconds - Avg Loss: 2.2012, Avg Acc: 0.1996
Evaluating on 10000 samples...
Test Loss: 2.0603, Test Accuracy: 0.2507

Epoch 2/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.9955, Acc: 0.2904
  Epoch completed in 17.66 seconds - Avg Loss: 1.9955, Avg Acc: 0.2904
Evaluating on 10000 samples...
Test Loss: 1.9414, Test Accuracy: 0.3037

Epoch 3/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.8874, Acc: 0.3289
  Epoch completed in 17.87 seconds - Avg Loss: 1.8874, Avg Acc: 0.3289
Evaluating on 10000 samples...
Test Loss: 1.8421, Test Accuracy: 0.3431

Epoch 4/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.8147, Acc: 0.3534
  Epoch completed in 18.06 seconds - Avg Loss: 1.8147, Avg Acc: 0.3534
Evaluating on 10000 samples...
Test Loss: 1.7804, Test Accuracy: 0.3676

Epoch 5/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.7606, Acc: 0.3714
  Epoch completed in 18.20 seconds - Avg Loss: 1.7606, Avg Acc: 0.3714
Evaluating on 10000 samples...
Test Loss: 1.7392, Test Accuracy: 0.3787

Epoch 6/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.7180, Acc: 0.3873
  Epoch completed in 18.33 seconds - Avg Loss: 1.7180, Avg Acc: 0.3873
Evaluating on 10000 samples...
Test Loss: 1.7029, Test Accuracy: 0.3895

Epoch 7/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.6822, Acc: 0.4005
  Epoch completed in 18.48 seconds - Avg Loss: 1.6822, Avg Acc: 0.4005
Evaluating on 10000 samples...
Test Loss: 1.6722, Test Accuracy: 0.4012

Epoch 8/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.6509, Acc: 0.4121
  Epoch completed in 18.64 seconds - Avg Loss: 1.6509, Avg Acc: 0.4121
Evaluating on 10000 samples...
Test Loss: 1.6441, Test Accuracy: 0.4108

Epoch 9/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.6224, Acc: 0.4230
  Epoch completed in 18.74 seconds - Avg Loss: 1.6224, Avg Acc: 0.4230
Evaluating on 10000 samples...
Test Loss: 1.6166, Test Accuracy: 0.4198

Epoch 10/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.5958, Acc: 0.4320
  Epoch completed in 18.76 seconds - Avg Loss: 1.5958, Avg Acc: 0.4320
Evaluating on 10000 samples...
Test Loss: 1.5914, Test Accuracy: 0.4316

Epoch 11/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.5707, Acc: 0.4422
  Epoch completed in 18.73 seconds - Avg Loss: 1.5707, Avg Acc: 0.4422
Evaluating on 10000 samples...
Test Loss: 1.5691, Test Accuracy: 0.4394

Epoch 12/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.5469, Acc: 0.4508
  Epoch completed in 18.70 seconds - Avg Loss: 1.5469, Avg Acc: 0.4508
Evaluating on 10000 samples...
Test Loss: 1.5467, Test Accuracy: 0.4484

Epoch 13/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.5245, Acc: 0.4589
  Epoch completed in 18.72 seconds - Avg Loss: 1.5245, Avg Acc: 0.4589
Evaluating on 10000 samples...
Test Loss: 1.5267, Test Accuracy: 0.4554

Epoch 14/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.5031, Acc: 0.4664
  Epoch completed in 18.75 seconds - Avg Loss: 1.5031, Avg Acc: 0.4664
Evaluating on 10000 samples...
Test Loss: 1.5091, Test Accuracy: 0.4615

Epoch 15/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.4827, Acc: 0.4740
  Epoch completed in 18.74 seconds - Avg Loss: 1.4827, Avg Acc: 0.4740
Evaluating on 10000 samples...
Test Loss: 1.4918, Test Accuracy: 0.4691

Epoch 16/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.4632, Acc: 0.4809
  Epoch completed in 18.75 seconds - Avg Loss: 1.4632, Avg Acc: 0.4809
Evaluating on 10000 samples...
Test Loss: 1.4751, Test Accuracy: 0.4746

Epoch 17/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.4443, Acc: 0.4874
  Epoch completed in 18.74 seconds - Avg Loss: 1.4443, Avg Acc: 0.4874
Evaluating on 10000 samples...
Test Loss: 1.4606, Test Accuracy: 0.4802

Epoch 18/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.4264, Acc: 0.4944
  Epoch completed in 18.74 seconds - Avg Loss: 1.4264, Avg Acc: 0.4944
Evaluating on 10000 samples...
Test Loss: 1.4458, Test Accuracy: 0.4852

Epoch 19/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.4090, Acc: 0.5011
  Epoch completed in 18.74 seconds - Avg Loss: 1.4090, Avg Acc: 0.5011
Evaluating on 10000 samples...
Test Loss: 1.4337, Test Accuracy: 0.4915

Epoch 20/20:
  Progress: [========================================] 781/781 (100.0%) - Loss: 1.3922, Acc: 0.5075
  Epoch completed in 18.74 seconds - Avg Loss: 1.3922, Avg Acc: 0.5075
Evaluating on 10000 samples...
Test Loss: 1.4227, Test Accuracy: 0.4998

=== Training Complete ===
Total training time: 369.60 seconds
Average time per epoch: 18.48 seconds
```

## Feature extract

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
Saving features to ../extracted_features/train_features_v3.bin...
Features saved: 50000 samples x 8192 features

=== Extracting Test Features ===
Extracting features from 10000 samples...
  Processed 157/157 batches (100.0%)
Feature extraction complete: 10000 samples x 8192 features
Saving features to ../extracted_features/test_features_v3.bin...
Features saved: 10000 samples x 8192 features

=== Training Classifier on Extracted Features ===
Train features: (50000, 8192)
Test features: (10000, 8192)


Training classifier...
Samples: 50000, Features: 8192, Classes: 10
Epochs: 20, Learning rate: 0.0010

Epoch 1/20 - Loss: 1.6337 - Time: 9.73s
Epoch 2/20 - Loss: 1.4978 - Time: 9.04s
Epoch 3/20 - Loss: 1.4565 - Time: 9.59s
Epoch 4/20 - Loss: 1.4309 - Time: 9.65s
Epoch 5/20 - Loss: 1.4121 - Time: 9.71s
Epoch 6/20 - Loss: 1.3970 - Time: 9.08s
Epoch 7/20 - Loss: 1.3844 - Time: 9.69s
Epoch 8/20 - Loss: 1.3735 - Time: 9.80s
Epoch 9/20 - Loss: 1.3640 - Time: 9.68s
Epoch 10/20 - Loss: 1.3554 - Time: 9.45s
Epoch 11/20 - Loss: 1.3476 - Time: 9.36s
Epoch 12/20 - Loss: 1.3404 - Time: 9.59s
Epoch 13/20 - Loss: 1.3338 - Time: 9.53s
Epoch 14/20 - Loss: 1.3277 - Time: 9.71s
Epoch 15/20 - Loss: 1.3220 - Time: 9.20s
Epoch 16/20 - Loss: 1.3166 - Time: 9.61s
Epoch 17/20 - Loss: 1.3115 - Time: 9.62s
Epoch 18/20 - Loss: 1.3067 - Time: 9.63s
Epoch 19/20 - Loss: 1.3022 - Time: 9.05s
Epoch 20/20 - Loss: 1.2978 - Time: 9.69s

=== Evaluation ===
Training Accuracy: 0.5481 (54.81%)
Test Accuracy: 0.5141 (51.41%)
```
## SVM Eval
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
Data transfer to GPU: 7.21 seconds

Predicting on GPU...
✓ Prediction completed in 16.19 seconds
  Average time per sample: 1.62 ms

============================================================
Evaluation Results
============================================================
Test Accuracy: 46.88%

Per-Class Accuracy:
  airplane     (Class 0): 59.30%
  automobile   (Class 1): 53.10%
  bird         (Class 2): 35.50%
  cat          (Class 3): 32.30%
  deer         (Class 4): 40.50%
  dog          (Class 5): 37.60%
  frog         (Class 6): 48.70%
  horse        (Class 7): 52.10%
  ship         (Class 8): 57.50%
  truck        (Class 9): 52.20%

Detailed Classification Report:
              precision    recall  f1-score   support

    airplane     0.5161    0.5930    0.5519      1000
  automobile     0.5803    0.5310    0.5546      1000
        bird     0.3484    0.3550    0.3517      1000
         cat     0.3103    0.3230    0.3165      1000
        deer     0.4210    0.4050    0.4128      1000
         dog     0.3929    0.3760    0.3843      1000
        frog     0.5047    0.4870    0.4957      1000
       horse     0.5273    0.5210    0.5241      1000
        ship     0.5861    0.5750    0.5805      1000
       truck     0.5103    0.5220    0.5161      1000

    accuracy                         0.4688     10000
   macro avg     0.4697    0.4688    0.4688     10000
weighted avg     0.4697    0.4688    0.4688     10000


Confusion Matrix (10x10):
[[593  24  67  40  31  19  17  23 125  61]
 [ 59 531  14  33  11  17  31  34  71 199]
 [118  20 355  83 135  87  92  69  26  15]
 [ 39  25  97 323  74 182 102  75  36  47]
 [ 39  17 138  84 405  70 110 106  18  13]
 [ 25  29 101 218  71 376  73  74  16  17]
 [ 11  26 117 110 110  70 487  31  13  25]
 [ 42  18  81  62  89  92  26 521  22  47]
 [160  58  25  36  21  21  11  16 575  77]
 [ 63 167  24  52  15  23  16  39  79 522]]

✓ Confusion matrix saved to: ./models/confusion_matrix_cuml.png
✓ Results saved to: ./models/evaluation_results_cuml.txt

============================================================
Evaluation Summary
============================================================
✓ Test Accuracy: 46.88%
✓ Prediction Time: 16.19 seconds
✓ Results saved successfully!
============================================================
```