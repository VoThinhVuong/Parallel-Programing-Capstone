"""
Train SVM using ThunderSVM command-line interface
Uses thundersvm-train executable instead of Python library
"""

import numpy as np
import struct
import os
import subprocess
import argparse

def load_features(features_path, labels_path):
    """Load features and labels from binary files."""
    print(f"Loading features from {features_path}...")
    with open(features_path, 'rb') as f:
        num_samples = struct.unpack('i', f.read(4))[0]
        file_size = os.path.getsize(features_path)
        feature_bytes = file_size - 4
        feature_dim = feature_bytes // (num_samples * 4)
        features = np.fromfile(f, dtype=np.float32, count=num_samples * feature_dim)
        features = features.reshape(num_samples, feature_dim)
    
    print(f"Loading labels from {labels_path}...")
    with open(labels_path, 'rb') as f:
        num_samples_labels = struct.unpack('i', f.read(4))[0]
        labels = np.fromfile(f, dtype=np.uint8, count=num_samples_labels)
    
    assert num_samples == num_samples_labels
    print(f"Loaded {num_samples} samples with {feature_dim} features")
    return features, labels

def save_libsvm_format(features, labels, output_path):
    """Save data in LIBSVM format for ThunderSVM."""
    print(f"Saving LIBSVM format to {output_path}...")
    with open(output_path, 'w') as f:
        for i in range(len(labels)):
            # Write label
            f.write(str(int(labels[i])))
            # Write features in sparse format (feature_id:value)
            for j in range(features.shape[1]):
                if features[i, j] != 0:  # Only write non-zero features
                    f.write(f" {j+1}:{features[i, j]}")
            f.write('\n')
    print(f"✓ Saved {len(labels)} samples")

def train_svm_cli(train_file, model_file, kernel='linear', C=10.0, gamma=None):
    """Train SVM using thundersvm-train executable."""
    
    # Build command
    cmd = ['./thundersvm-train']
    
    # Kernel type: -t 0=linear, 2=rbf
    if kernel == 'linear':
        cmd.extend(['-t', '0'])
    elif kernel == 'rbf':
        cmd.extend(['-t', '2'])
        if gamma:
            cmd.extend(['-g', str(gamma)])
    
    # Regularization parameter
    cmd.extend(['-c', str(C)])
    
    # Input and output files
    cmd.extend([train_file, model_file])
    
    print(f"\nTraining SVM with command:")
    print(f"  {' '.join(cmd)}\n")
    
    # Run training
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with return code {result.returncode}")
    
    print(f"✓ Model saved to {model_file}")

def main():
    parser = argparse.ArgumentParser(description='Train SVM using ThunderSVM CLI')
    parser.add_argument('gpu_version', nargs='?', default='naive', choices=['naive', 'shared', 'v3'],
                        help='GPU version (naive, shared, or v3)')
    parser.add_argument('--subset', type=int, default=None,
                        help='Train on subset of samples')
    parser.add_argument('--kernel', type=str, default='linear', choices=['linear', 'rbf'],
                        help='Kernel type')
    parser.add_argument('--C', type=float, default=10.0,
                        help='Regularization parameter')
    parser.add_argument('--gamma', type=float, default=None,
                        help='Gamma for RBF kernel (default: 1/n_features)')
    args = parser.parse_args()
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    features_dir = os.path.join(parent_dir, 'extracted_features')
    
    suffix_map = {'naive': '_naive', 'shared': '_shared', 'v3': '_v3'}
    suffix = suffix_map[args.gpu_version]
    
    train_features_path = os.path.join(features_dir, f'train_features{suffix}.bin')
    train_labels_path = os.path.join(features_dir, 'train_labels.bin')
    
    # Check files exist
    if not os.path.exists(train_features_path):
        print(f"ERROR: Features not found at {train_features_path}")
        return
    
    # Load data
    train_features, train_labels = load_features(train_features_path, train_labels_path)
    
    # Apply subset if requested
    if args.subset and args.subset < len(train_features):
        print(f"\nUsing subset: {args.subset} of {len(train_features)} samples")
        np.random.seed(42)
        indices = np.random.choice(len(train_features), args.subset, replace=False)
        train_features = train_features[indices]
        train_labels = train_labels[indices]
    
    # Memory warning for RBF
    if args.kernel == 'rbf':
        n_samples = len(train_features)
        kernel_matrix_gb = (n_samples * n_samples * 4) / (1024**3)
        print(f"\n⚠️  RBF kernel matrix size: ~{kernel_matrix_gb:.2f} GB")
        if kernel_matrix_gb > 2:
            print("   Consider using --subset or --kernel linear")
    
    # Save in LIBSVM format
    train_file = os.path.join(base_dir, f'train_data{suffix}.txt')
    save_libsvm_format(train_features, train_labels, train_file)
    
    # Model output path
    model_name = f'svm_model{suffix}'
    if args.subset:
        model_name += f'_subset{args.subset}'
    if args.kernel != 'linear':
        model_name += f'_{args.kernel}'
    model_file = os.path.join(base_dir, f'{model_name}.model')
    
    # Calculate gamma for RBF
    gamma = args.gamma
    if args.kernel == 'rbf' and gamma is None:
        gamma = 1.0 / train_features.shape[1]
        print(f"Using gamma = 1/n_features = {gamma}")
    
    # Train
    print("\n" + "="*60)
    print(f"Training SVM ({args.kernel} kernel, C={args.C})")
    print("="*60)
    
    try:
        train_svm_cli(train_file, model_file, 
                      kernel=args.kernel, C=args.C, gamma=gamma)
    except FileNotFoundError:
        print("\n✗ ERROR: thundersvm-train executable not found!")
        print("Please run 'make' in the SVM folder first.")
        return
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        return
    
    print("\n" + "="*60)
    print("✓ Training Complete!")
    print("="*60)
    print(f"Model: {model_file}")
    print(f"Training data: {train_file}")
    print(f"\nNext: python evaluate_svm_cli.py {args.gpu_version} --model {model_name}.model")

if __name__ == '__main__':
    main()
