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


def train_svm(train_features, train_labels, C=10.0, gamma='auto'):
    """
    Train SVM with RBF kernel using ThunderSVM.
    
    Args:
        train_features: Training features (N x D)
        train_labels: Training labels (N,)
        C: Regularization parameter (default: 10.0)
        gamma: Kernel coefficient (default: 'auto' = 1/n_features)
    
    Returns:
        Trained SVM model
    """
    print("\n" + "="*60)
    print("Training SVM with ThunderSVM")
    print("="*60)
    print(f"Hyperparameters:")
    print(f"  Kernel: RBF")
    print(f"  C: {C}")
    print(f"  gamma: {gamma}")
    print(f"\nTraining data:")
    print(f"  Samples: {train_features.shape[0]}")
    print(f"  Features: {train_features.shape[1]}")
    print(f"  Classes: {len(np.unique(train_labels))}")
    
    # Create SVM classifier with ThunderSVM
    svm = SVC(
        kernel='rbf',
        C=C,
        gamma=gamma,
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
    # Paths to feature files (from GPU_naive folder)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    gpu_naive_dir = os.path.join(parent_dir, 'GPU_naive')
    
    train_features_path = os.path.join(gpu_naive_dir, 'train_features.bin')
    train_labels_path = os.path.join(gpu_naive_dir, 'train_labels.bin')
    model_output_path = os.path.join(base_dir, 'svm_model.pkl')
    
    # Check if files exist
    if not os.path.exists(train_features_path):
        print(f"ERROR: Training features not found at {train_features_path}")
        print("Please run the feature extraction in GPU_naive folder first.")
        return
    
    if not os.path.exists(train_labels_path):
        print(f"ERROR: Training labels not found at {train_labels_path}")
        return
    
    # Load training data
    train_features, train_labels = load_features(train_features_path, train_labels_path)
    
    # Train SVM with specified hyperparameters
    svm_model = train_svm(
        train_features, 
        train_labels, 
        C=10.0,          # Regularization parameter
        gamma='auto'     # Kernel coefficient (auto = 1/n_features)
    )
    
    # Save the trained model
    save_model(svm_model, model_output_path)
    
    print("\n" + "="*60)
    print("SVM Training Complete!")
    print("="*60)
    print(f"Model saved to: {model_output_path}")
    print(f"\nNext step: Run evaluate_svm.py to test the model")


if __name__ == '__main__':
    main()
