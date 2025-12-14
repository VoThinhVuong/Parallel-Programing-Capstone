"""
Step 4: Train SVM using cuML (GPU-accelerated)
- Input: train_features + labels from extracted_features folder
- Kernel: RBF (Radial Basis Function)
- Hyperparameters: C=10, gamma=auto
- Output: trained SVM model
"""

import numpy as np
import cupy as cp
import cudf
import struct
import os
import pickle
import time
import argparse
from cuml.svm import SVC
import warnings
warnings.filterwarnings('ignore')


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
        # Convert to int32 for compatibility with cuML
        labels = labels.astype(np.int32)
    
    assert num_samples == num_samples_labels, "Mismatch between features and labels count!"
    
    print(f"Loaded features shape: {features.shape}")
    print(f"Loaded labels shape: {labels.shape}")
    print(f"Label range: {labels.min()} to {labels.max()}")
    
    return features, labels


def train_svm_cuml(train_features, train_labels, C=10.0, gamma='auto', kernel='rbf'):
    """
    Train SVM with RBF kernel using cuML (GPU-accelerated).
    
    Args:
        train_features: Training features (N x D) as numpy array
        train_labels: Training labels (N,) as numpy array
        C: Regularization parameter (default: 10.0)
        gamma: Kernel coefficient (default: 'auto' = 1/n_features)
        kernel: Kernel type ('rbf' or 'linear')
    
    Returns:
        Trained cuML SVM model
    """
    print("\n" + "="*60)
    print("Training SVM with cuML (GPU-accelerated)")
    print("="*60)
    print(f"Hyperparameters:")
    print(f"  Kernel: {kernel.upper()}")
    print(f"  C: {C}")
    if kernel == 'rbf':
        if gamma == 'auto':
            actual_gamma = 1.0 / train_features.shape[1]
            print(f"  gamma: auto (1/n_features = {actual_gamma:.6f})")
        else:
            print(f"  gamma: {gamma}")
    print(f"\nTraining data:")
    print(f"  Samples: {train_features.shape[0]}")
    print(f"  Features: {train_features.shape[1]}")
    print(f"  Classes: {len(np.unique(train_labels))}")
    
    # Convert numpy arrays to cuDF for GPU processing
    print("\nTransferring data to GPU...")
    start_transfer = time.time()
    
    # cuML SVC expects cuDF Series for labels and cuDF DataFrame for features
    X_train_cudf = cudf.DataFrame(train_features)
    y_train_cudf = cudf.Series(train_labels)
    
    transfer_time = time.time() - start_transfer
    print(f"Data transfer to GPU: {transfer_time:.2f} seconds")
    
    # Create SVM classifier with cuML
    print("\nInitializing cuML SVC...")
    svm = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma if kernel == 'rbf' else 'auto',
        cache_size=2000.0,  # Cache size in MB for kernel computation
        max_iter=-1,  # No iteration limit
        nochange_steps=1000,  # Early stopping criterion
        tol=0.001,  # Tolerance for stopping criterion
        verbose=True
    )
    
    # Train the model
    print("\nTraining SVM on GPU...")
    print("-" * 60)
    start_time = time.time()
    
    svm.fit(X_train_cudf, y_train_cudf)
    
    training_time = time.time() - start_time
    print("-" * 60)
    print(f"\n✓ Training completed in {training_time:.2f} seconds")
    
    # Model statistics
    print(f"\nModel statistics:")
    print(f"  Support vectors: {svm.n_support_}")
    print(f"  Number of classes: {len(svm.classes_)}")
    print(f"  Classes: {svm.classes_}")
    
    return svm, training_time


def save_model(model, output_path):
    """Save the trained cuML SVM model to disk."""
    print(f"\nSaving model to {output_path}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # Convert to MB
    print(f"✓ Model saved successfully! ({file_size:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description='Train SVM using cuML on extracted CNN features')
    
    # Input paths
    parser.add_argument('--train-features', type=str, 
                        default='../extracted_features/train_features.bin',
                        help='Path to training features binary file')
    parser.add_argument('--train-labels', type=str,
                        default='../extracted_features/train_labels.bin',
                        help='Path to training labels binary file')
    
    # Output path
    parser.add_argument('--output', type=str,
                        default='./models/svm_model.pkl',
                        help='Path to save trained model')
    
    # Hyperparameters
    parser.add_argument('--C', type=float, default=10.0,
                        help='SVM regularization parameter (default: 10.0)')
    parser.add_argument('--gamma', type=str, default='auto',
                        help='Kernel coefficient (default: auto)')
    parser.add_argument('--kernel', type=str, default='rbf',
                        choices=['rbf', 'linear'],
                        help='Kernel type (default: rbf)')
    
    # Optional subset for testing
    parser.add_argument('--subset', type=int, default=None,
                        help='Use only first N samples for quick testing')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SVM Training with cuML (GPU-accelerated)")
    print("="*60)
    print(f"Configuration:")
    print(f"  Train features: {args.train_features}")
    print(f"  Train labels: {args.train_labels}")
    print(f"  Output model: {args.output}")
    print(f"  Kernel: {args.kernel}")
    print(f"  C: {args.C}")
    print(f"  Gamma: {args.gamma}")
    if args.subset:
        print(f"  Subset: {args.subset} samples (testing mode)")
    print()
    
    # Load training data
    train_features, train_labels = load_features(args.train_features, args.train_labels)
    
    # Use subset if specified
    if args.subset:
        print(f"\n⚠️  Using subset of {args.subset} samples for testing")
        train_features = train_features[:args.subset]
        train_labels = train_labels[:args.subset]
    
    # Train SVM
    try:
        # Convert gamma if it's not 'auto'
        gamma = args.gamma if args.gamma == 'auto' else float(args.gamma)
        
        svm, training_time = train_svm_cuml(
            train_features,
            train_labels,
            C=args.C,
            gamma=gamma,
            kernel=args.kernel
        )
        
        # Save model
        save_model(svm, args.output)
        
        # Summary
        print("\n" + "="*60)
        print("Training Summary")
        print("="*60)
        print(f"✓ Training time: {training_time:.2f} seconds")
        print(f"✓ Model saved to: {args.output}")
        print(f"✓ Ready for evaluation!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
