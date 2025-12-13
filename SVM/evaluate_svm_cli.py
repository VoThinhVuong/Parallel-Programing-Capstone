"""
Evaluate SVM using ThunderSVM command-line interface
Uses thundersvm-predict executable
"""

import numpy as np
import struct
import os
import subprocess
import argparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
    """Save data in LIBSVM format."""
    print(f"Saving test data to {output_path}...")
    with open(output_path, 'w') as f:
        for i in range(len(labels)):
            f.write(str(int(labels[i])))
            for j in range(features.shape[1]):
                if features[i, j] != 0:
                    f.write(f" {j+1}:{features[i, j]}")
            f.write('\n')
    print(f"✓ Saved {len(labels)} samples")

def predict_svm_cli(test_file, model_file, output_file):
    """Predict using thundersvm-predict executable."""
    
    cmd = ['./thundersvm-predict', test_file, model_file, output_file]
    
    print(f"\nPredicting with command:")
    print(f"  {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        raise RuntimeError(f"Prediction failed with return code {result.returncode}")

def load_predictions(output_file):
    """Load predictions from ThunderSVM output."""
    predictions = []
    with open(output_file, 'r') as f:
        for line in f:
            predictions.append(int(float(line.strip())))
    return np.array(predictions)

def plot_confusion_matrix(cm, output_path, class_names):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate SVM using ThunderSVM CLI')
    parser.add_argument('gpu_version', nargs='?', default='naive', choices=['naive', 'shared', 'v3'],
                        help='GPU version (naive, shared, or v3)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model filename (e.g., svm_model_v3.model)')
    args = parser.parse_args()
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    features_dir = os.path.join(parent_dir, 'extracted_features')
    
    suffix_map = {'naive': '_naive', 'shared': '_shared', 'v3': '_v3'}
    suffix = suffix_map[args.gpu_version]
    
    # Model path
    if args.model:
        model_file = os.path.join(base_dir, args.model)
    else:
        model_file = os.path.join(base_dir, f'svm_model{suffix}.model')
    
    if not os.path.exists(model_file):
        print(f"ERROR: Model not found at {model_file}")
        print(f"Please train the model first: python train_svm_cli.py {args.gpu_version}")
        return
    
    # Test data paths
    test_features_path = os.path.join(features_dir, f'test_features{suffix}.bin')
    test_labels_path = os.path.join(features_dir, 'test_labels.bin')
    
    if not os.path.exists(test_features_path):
        print(f"ERROR: Test features not found at {test_features_path}")
        return
    
    # Load test data
    test_features, test_labels = load_features(test_features_path, test_labels_path)
    
    # Save test data in LIBSVM format
    test_file = os.path.join(base_dir, f'test_data{suffix}.txt')
    save_libsvm_format(test_features, test_labels, test_file)
    
    # Output file for predictions
    output_file = os.path.join(base_dir, f'predictions{suffix}.txt')
    
    # Predict
    print("\n" + "="*60)
    print("Evaluating SVM")
    print("="*60)
    
    try:
        predict_svm_cli(test_file, model_file, output_file)
    except FileNotFoundError:
        print("\n✗ ERROR: thundersvm-predict executable not found!")
        print("Please run 'make' in the SVM folder first.")
        return
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        return
    
    # Load predictions
    predictions = load_predictions(output_file)
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions)
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Correct: {int(accuracy * len(test_labels))}/{len(test_labels)}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    
    # Save results
    results_file = os.path.join(base_dir, f'evaluation_results{suffix}.txt')
    with open(results_file, 'w') as f:
        f.write(f"Test Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"Correct: {int(accuracy * len(test_labels))}/{len(test_labels)}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(test_labels, predictions, target_names=class_names))
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
    print(f"\n✓ Results saved to {results_file}")
    
    # Plot confusion matrix
    cm_plot_path = os.path.join(base_dir, f'confusion_matrix{suffix}.png')
    plot_confusion_matrix(cm, cm_plot_path, class_names)
    
    print("\n" + "="*60)
    print("✓ Evaluation Complete!")
    print("="*60)

if __name__ == '__main__':
    main()
