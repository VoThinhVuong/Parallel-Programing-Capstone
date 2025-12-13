"""
Step 4: Train SVM using ThunderSVM
- Input: train_features + labels from GPU_naive folder
- Kernel: RBF (Radial Basis Function)
- Hyperparameters: C=10, gamma=auto
- Output: trained SVM model
"""

import numpy as np
import struct
import os
from thundersvm_lib import SVC
import pickle
import time

def load_features(features_path, labels_path):
    """
    Load features and labels from binary files.
    
    Features format: 4 bytes (num_samples) + float32 array
    Labels format: 4 bytes (num_samples) + uint8 array
    """
    print(f"Loading features from {features_path}...")
    with open(features_path, 'rb') as f:
        # Read number of samples (first 4 bytes)
        num_samples = struct.unpack('i', f.read(4))[0]
        print(f"  Number of samples: {num_samples}")
        
        # Calculate feature dimension from file size
        file_size = os.path.getsize(features_path)
        feature_bytes = file_size - 4  # Subtract header
        feature_dim = feature_bytes // (num_samples * 4)  # 4 bytes per float32
        print(f"  Feature dimension: {feature_dim}")
        
        # Read features
        features = np.fromfile(f, dtype=np.float32, count=num_samples * feature_dim)
        features = features.reshape(num_samples, feature_dim)
    
    print(f"Loading labels from {labels_path}...")
    with open(labels_path, 'rb') as f:
        # Read number of samples (first 4 bytes)
        num_samples_labels = struct.unpack('i', f.read(4))[0]
        print(f"  Number of labels: {num_samples_labels}")
        
        # Read labels (stored as uint8, 1 byte per label)
        labels = np.fromfile(f, dtype=np.uint8, count=num_samples_labels)
        # Convert to int32 for compatibility with sklearn
        labels = labels.astype(np.int32)
    
    assert num_samples == num_samples_labels, "Mismatch between features and labels count!"
    
    print(f"Loaded features shape: {features.shape}")
    print(f"Loaded labels shape: {labels.shape}")
    print(f"Label range: {labels.min()} to {labels.max()}")
    
    return features, labels


def train_svm(train_features, train_labels, C=10.0, gamma='auto', kernel='rbf'):
    """
    Train SVM with RBF kernel using ThunderSVM.
    
    Args:
        train_features: Training features (N x D)
        train_labels: Training labels (N,)
        C: Regularization parameter (default: 10.0)
        gamma: Kernel coefficient (default: 'auto' = 1/n_features)
        kernel: Kernel type ('rbf' or 'linear')
    
    Returns:
        Trained SVM model
    """
    print("\n" + "="*60)
    print("Training SVM with ThunderSVM")
    print("="*60)
    print(f"Hyperparameters:")
    print(f"  Kernel: {kernel.upper()}")
    print(f"  C: {C}")
    if kernel == 'rbf':
        print(f"  gamma: {gamma}")
    print(f"\nTraining data:")
    print(f"  Samples: {train_features.shape[0]}")
    print(f"  Features: {train_features.shape[1]}")
    print(f"  Classes: {len(np.unique(train_labels))}")
    
    # Memory estimation for RBF kernel
    if kernel == 'rbf':
        n_samples = train_features.shape[0]
        kernel_matrix_gb = (n_samples * n_samples * 4) / (1024**3)
        print(f"\n⚠️  RBF kernel matrix size: ~{kernel_matrix_gb:.2f} GB")
        if kernel_matrix_gb > 2:
            print(f"   WARNING: Large memory usage! Consider:")
            print(f"   - Using --subset <num> to train on fewer samples")
            print(f"   - Using --kernel linear for lower memory")
    
    # Create SVM classifier with ThunderSVM
    svm = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma if kernel == 'rbf' else 'auto',
        verbose=True,
        random_state=42
    )
    
    # Train the model
    print("\nTraining started...")
    start_time = time.time()
    svm.fit(train_features, train_labels)
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Calculate training accuracy
    train_pred = svm.predict(train_features)
    train_accuracy = np.mean(train_pred == train_labels) * 100
    print(f"Training accuracy: {train_accuracy:.2f}%")
    
    return svm


def save_model(model, model_path):
    """Save the trained SVM model to disk."""
    print(f"\nSaving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully!")


def main():
    # Paths to feature files (from extracted_features folder)
    import sys
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train SVM on CNN features')
    parser.add_argument('gpu_version', nargs='?', default='naive', choices=['naive', 'shared', 'v3'],
                        help='GPU version to use (naive, shared, or v3)')
    parser.add_argument('--subset', type=int, default=None,
                        help='Train on subset of samples (e.g., 10000) to reduce memory usage')
    parser.add_argument('--kernel', type=str, default='rbf', choices=['rbf', 'linear'],
                        help='SVM kernel type (rbf=high accuracy/high memory, linear=lower memory)')
    parser.add_argument('--C', type=float, default=10.0,
                        help='SVM regularization parameter (default: 10.0)')
    args = parser.parse_args()
    
    gpu_version = args.gpu_version
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    features_dir = os.path.join(parent_dir, 'extracted_features')
    
    if gpu_version == 'naive':
        suffix = '_naive'
    elif gpu_version == 'shared':
        suffix = '_shared'
    else:  # v3
        suffix = '_v3'
    
    train_features_path = os.path.join(features_dir, f'train_features{suffix}.bin')
    train_labels_path = os.path.join(features_dir, 'train_labels.bin')
    
    # Add subset/kernel info to model name if using non-default settings
    model_name = f'svm_model{suffix}'
    if args.subset:
        model_name += f'_subset{args.subset}'
    if args.kernel != 'rbf':
        model_name += f'_{args.kernel}'
    model_output_path = os.path.join(base_dir, f'{model_name}.pkl')
    
    print(f"Using GPU version: {gpu_version}")
    if args.subset:
        print(f"Training on subset: {args.subset} samples")
    if args.kernel != 'rbf':
        print(f"Using {args.kernel} kernel")
    print(f"Feature files will be read from: {features_dir}")
    print(f"Model will be saved as: {model_name}.pkl\n")
    
    # Check if files exist
    if not os.path.exists(train_features_path):
        print(f"ERROR: Training features not found at {train_features_path}")
        folder_map = {'naive': 'GPU_naive', 'shared': 'GPU_v2_shared_mem', 'v3': 'GPU_v3_optimized_gradient'}
        print(f"Please run the feature extraction in {folder_map.get(gpu_version, 'GPU')} folder first.")
        print(f"Usage: python train_svm.py [naive|shared|v3]")
        return
    
    if not os.path.exists(train_labels_path):
        print(f"ERROR: Training labels not found at {train_labels_path}")
        return
    
    # Load training data
    train_features, train_labels = load_features(train_features_path, train_labels_path)
    
    # Use subset if specified (to reduce memory usage)
    if args.subset and args.subset < len(train_features):
        print(f"\n⚠️  Using subset: {args.subset} of {len(train_features)} samples")
        # Randomly sample with fixed seed for reproducibility
        np.random.seed(42)
        indices = np.random.choice(len(train_features), args.subset, replace=False)
        train_features = train_features[indices]
        train_labels = train_labels[indices]
        print(f"Subset shape: {train_features.shape}")
    
    # Train SVM with specified hyperparameters
    svm_model = train_svm(
        train_features, 
        train_labels, 
        C=args.C,           # Regularization parameter
        gamma='auto',       # Kernel coeffici{gpu_version} --model {model_name}.pkl to test the model")
        kernel=args.kernel  # Kernel type
    )
    
    print(f"\nUsage tips:")
    print(f"  - For lower memory: python train_svm.py {gpu_version} --subset 15000")
    print(f"  - For fastest training: python train_svm.py {gpu_version} --kernel linear")
    print(f"  - Combination: python train_svm.py {gpu_version} --subset 15000 --kernel lineareatures")
    
    # Save the trained model
    save_model(svm_model, model_output_path)
    
    print("\n" + "="*60)
    print("SVM Training Complete!")
    print("="*60)
    print(f"Model saved to: {model_output_path}")
    print(f"\nNext step: Run evaluate_svm.py to test the model")


if __name__ == '__main__':
    main()
