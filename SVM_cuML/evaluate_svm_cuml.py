"""
Step 5: Evaluate SVM using cuML (GPU-accelerated)
- Predict on test_features using trained SVM
- Calculate accuracy and confusion matrix
- Expected accuracy: 60-65%
- Compare with baseline methods
"""

import numpy as np
import cupy as cp
import cudf
import struct
import os
import pickle
import time
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
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
        # Convert to int32 for compatibility
        labels = labels.astype(np.int32)
    
    assert num_samples == num_samples_labels, "Mismatch between features and labels count!"
    
    print(f"Loaded features shape: {features.shape}")
    print(f"Loaded labels shape: {labels.shape}")
    
    return features, labels


def load_model(model_path):
    """Load the trained cuML SVM model from disk."""
    print(f"Loading cuML model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded successfully!")
    return model


def evaluate_svm_cuml(model, test_features, test_labels):
    """
    Evaluate the cuML SVM model on test data (GPU-accelerated).
    
    Args:
        model: Trained cuML SVM model
        test_features: Test features (N x D) as numpy array
        test_labels: True test labels (N,) as numpy array
    
    Returns:
        predictions: Predicted labels (numpy array)
        accuracy: Test accuracy
        conf_matrix: Confusion matrix
        prediction_time: Time taken for prediction
    """
    print("\n" + "="*60)
    print("Evaluating SVM on Test Data (GPU-accelerated)")
    print("="*60)
    print(f"Test samples: {test_features.shape[0]}")
    print(f"Test features: {test_features.shape[1]}")
    
    # Transfer data to GPU
    print("\nTransferring test data to GPU...")
    start_transfer = time.time()
    X_test_cudf = cudf.DataFrame(test_features)
    transfer_time = time.time() - start_transfer
    print(f"Data transfer to GPU: {transfer_time:.2f} seconds")
    
    # Make predictions on GPU
    print("\nPredicting on GPU...")
    start_time = time.time()
    predictions_cudf = model.predict(X_test_cudf)
    
    # Convert predictions back to numpy for metrics calculation
    predictions = predictions_cudf.to_numpy()
    prediction_time = time.time() - start_time
    
    print(f"✓ Prediction completed in {prediction_time:.2f} seconds")
    print(f"  Average time per sample: {prediction_time/len(test_features)*1000:.2f} ms")
    
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions) * 100
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(test_labels, predictions)
    
    return predictions, accuracy, conf_matrix, prediction_time


def print_evaluation_results(test_labels, predictions, accuracy, conf_matrix):
    """Print detailed evaluation results."""
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    for i in range(10):
        class_mask = test_labels == i
        if class_mask.sum() > 0:
            class_acc = (predictions[class_mask] == i).sum() / class_mask.sum() * 100
            print(f"  {class_names[i]:12s} (Class {i}): {class_acc:.2f}%")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, predictions, 
                                target_names=class_names,
                                digits=4))
    
    # Confusion matrix summary
    print("\nConfusion Matrix (10x10):")
    print(conf_matrix)


def plot_confusion_matrix(conf_matrix, output_path='confusion_matrix_cuml.png'):
    """Plot and save confusion matrix heatmap."""
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - cuML SVM (CIFAR-10)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to: {output_path}")
    plt.close()


def save_results(results_dict, output_path='evaluation_results_cuml.txt'):
    """Save evaluation results to text file."""
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("cuML SVM Evaluation Results\n")
        f.write("="*60 + "\n\n")
        
        for key, value in results_dict.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"✓ Results saved to: {output_path}")


def compare_with_baseline(accuracy):
    """Compare results with baseline methods."""
    print("\n" + "="*60)
    print("Comparison with Baseline Methods")
    print("="*60)
    
    baselines = {
        "Random Guessing": 10.0,
        "Expected SVM (CPU)": 62.5,
        "Target Range (Low)": 60.0,
        "Target Range (High)": 65.0
    }
    
    print(f"{'Method':<25s} {'Accuracy':>10s} {'vs cuML':>15s}")
    print("-" * 60)
    
    for method, baseline_acc in baselines.items():
        diff = accuracy - baseline_acc
        symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(f"{method:<25s} {baseline_acc:>9.2f}% {symbol} {abs(diff):>7.2f}%")
    
    print("-" * 60)
    print(f"{'cuML SVM (GPU)':<25s} {accuracy:>9.2f}%")
    print("="*60)
    
    # Performance assessment
    if accuracy >= 60.0 and accuracy <= 65.0:
        print("\n✓ Result is within expected range (60-65%)")
    elif accuracy > 65.0:
        print("\n✓ Result exceeds expected range! Excellent performance!")
    else:
        print("\n⚠️  Result is below expected range (60-65%)")


def main():
    parser = argparse.ArgumentParser(description='Evaluate cuML SVM on test features')
    
    # Input paths
    parser.add_argument('--test-features', type=str,
                        default='../extracted_features/test_features.bin',
                        help='Path to test features binary file')
    parser.add_argument('--test-labels', type=str,
                        default='../extracted_features/test_labels.bin',
                        help='Path to test labels binary file')
    parser.add_argument('--model', type=str,
                        default='svm_model_cuml.pkl',
                        help='Path to trained cuML SVM model')
    
    # Output paths
    parser.add_argument('--output-matrix', type=str,
                        default='confusion_matrix_cuml.png',
                        help='Path to save confusion matrix plot')
    parser.add_argument('--output-results', type=str,
                        default='evaluation_results_cuml.txt',
                        help='Path to save evaluation results')
    
    # Optional subset for testing
    parser.add_argument('--subset', type=int, default=None,
                        help='Use only first N samples for quick testing')
    
    # Plotting option
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip confusion matrix plotting')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SVM Evaluation with cuML (GPU-accelerated)")
    print("="*60)
    print(f"Configuration:")
    print(f"  Test features: {args.test_features}")
    print(f"  Test labels: {args.test_labels}")
    print(f"  Model: {args.model}")
    if args.subset:
        print(f"  Subset: {args.subset} samples (testing mode)")
    print()
    
    try:
        # Load test data
        test_features, test_labels = load_features(args.test_features, args.test_labels)
        
        # Use subset if specified
        if args.subset:
            print(f"\n⚠️  Using subset of {args.subset} samples for testing")
            test_features = test_features[:args.subset]
            test_labels = test_labels[:args.subset]
        
        # Load model
        model = load_model(args.model)
        
        # Evaluate
        predictions, accuracy, conf_matrix, prediction_time = evaluate_svm_cuml(
            model, test_features, test_labels
        )
        
        # Print results
        print_evaluation_results(test_labels, predictions, accuracy, conf_matrix)
        
        # Compare with baselines
        compare_with_baseline(accuracy)
        
        # Plot confusion matrix
        if not args.no_plot:
            plot_confusion_matrix(conf_matrix, args.output_matrix)
        
        # Save results
        results_dict = {
            'Test Accuracy (%)': accuracy,
            'Test Samples': len(test_labels),
            'Prediction Time (s)': prediction_time,
            'Avg Time per Sample (ms)': prediction_time/len(test_features)*1000,
            'Model Path': args.model,
            'Confusion Matrix Shape': conf_matrix.shape
        }
        save_results(results_dict, args.output_results)
        
        # Final summary
        print("\n" + "="*60)
        print("Evaluation Summary")
        print("="*60)
        print(f"✓ Test Accuracy: {accuracy:.2f}%")
        print(f"✓ Prediction Time: {prediction_time:.2f} seconds")
        print(f"✓ Results saved successfully!")
        print("="*60)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: File not found - {e}")
        print("\nMake sure you have:")
        print("  1. Extracted features using GPU implementation")
        print("  2. Trained the SVM model using train_svm_cuml.py")
        return 1
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
