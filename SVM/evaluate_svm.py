"""
Step 5: Evaluate SVM
- Predict on test_features using trained SVM
- Calculate accuracy and confusion matrix
- Expected accuracy: 60-65%
"""

import numpy as np
import struct
import os
import pickle
import time
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


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
    
    return features, labels


def load_model(model_path):
    """Load the trained SVM model from disk."""
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
    return model


def evaluate_svm(model, test_features, test_labels):
    """
    Evaluate the SVM model on test data.
    
    Args:
        model: Trained SVM model
        test_features: Test features (N x D)
        test_labels: True test labels (N,)
    
    Returns:
        predictions: Predicted labels
        accuracy: Test accuracy
        conf_matrix: Confusion matrix
    """
    print("\n" + "="*60)
    print("Evaluating SVM on Test Data")
    print("="*60)
    print(f"Test samples: {test_features.shape[0]}")
    print(f"Test features: {test_features.shape[1]}")
    
    # Make predictions
    print("\nPredicting...")
    start_time = time.time()
    predictions = model.predict(test_features)
    prediction_time = time.time() - start_time
    
    print(f"Prediction completed in {prediction_time:.2f} seconds")
    print(f"Average time per sample: {prediction_time/len(test_features)*1000:.2f} ms")
    
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions) * 100
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(test_labels, predictions)
    
    return predictions, accuracy, conf_matrix


def print_results(test_labels, predictions, accuracy, conf_matrix):
    """Print detailed evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"Expected Range: 60-65%")
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Classification report
    print("\n" + "-"*60)
    print("Classification Report:")
    print("-"*60)
    report = classification_report(test_labels, predictions, 
                                   target_names=class_names,
                                   digits=4)
    print(report)
    
    # Per-class accuracy
    print("\n" + "-"*60)
    print("Per-Class Accuracy:")
    print("-"*60)
    for i, class_name in enumerate(class_names):
        class_mask = test_labels == i
        class_correct = np.sum((predictions[class_mask] == i))
        class_total = np.sum(class_mask)
        class_acc = (class_correct / class_total * 100) if class_total > 0 else 0
        print(f"{class_name:12s}: {class_acc:6.2f}% ({class_correct:4d}/{class_total:4d})")
    
    # Confusion matrix
    print("\n" + "-"*60)
    print("Confusion Matrix:")
    print("-"*60)
    print("Rows: True labels, Columns: Predicted labels")
    print(conf_matrix)


def plot_confusion_matrix(conf_matrix, output_path):
    """Plot and save confusion matrix visualization."""
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('SVM Confusion Matrix on CIFAR-10 Test Set', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix plot saved to: {output_path}")
    plt.close()


def save_results(accuracy, conf_matrix, predictions, output_path):
    """Save evaluation results to a text file."""
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("SVM EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Overall Test Accuracy: {accuracy:.2f}%\n")
        f.write(f"Expected Range: 60-65%\n\n")
        
        f.write("-"*60 + "\n")
        f.write("Per-Class Accuracy:\n")
        f.write("-"*60 + "\n")
        for i, class_name in enumerate(class_names):
            class_correct = conf_matrix[i, i]
            class_total = conf_matrix[i].sum()
            class_acc = (class_correct / class_total * 100) if class_total > 0 else 0
            f.write(f"{class_name:12s}: {class_acc:6.2f}% ({class_correct:4d}/{class_total:4d})\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("Confusion Matrix:\n")
        f.write("-"*60 + "\n")
        f.write("Rows: True labels, Columns: Predicted labels\n\n")
        
        # Header
        f.write("        ")
        for name in class_names:
            f.write(f"{name[:8]:>8s} ")
        f.write("\n")
        
        # Matrix rows
        for i, name in enumerate(class_names):
            f.write(f"{name[:8]:8s}")
            for j in range(len(class_names)):
                f.write(f"{conf_matrix[i, j]:8d} ")
            f.write("\n")
    
    print(f"Results saved to: {output_path}")


def main():
    # Paths to files
    import sys
    
    # Default to 'naive' version, but allow specifying 'shared' or 'v3' via command line
    gpu_version = 'naive'
    if len(sys.argv) > 1 and sys.argv[1] in ['naive', 'shared', 'v3']:
        gpu_version = sys.argv[1]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    features_dir = os.path.join(parent_dir, 'extracted_features')
    
    if gpu_version == 'naive':
        suffix = '_naive'
    elif gpu_version == 'shared':
        suffix = '_shared'
    else:  # v3
        suffix = '_v3'
    
    test_features_path = os.path.join(features_dir, f'test_features{suffix}.bin')
    test_labels_path = os.path.join(features_dir, 'test_labels.bin')
    model_path = os.path.join(base_dir, f'svm_model{suffix}.pkl')
    results_path = os.path.join(base_dir, f'evaluation_results{suffix}.txt')
    confusion_matrix_path = os.path.join(base_dir, f'confusion_matrix{suffix}.png')
    
    print(f"Using GPU version: {gpu_version}")
    print(f"Feature files will be read from: {features_dir}")
    print(f"Model file: svm_model{suffix}.pkl\n")
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print(f"Please run train_svm.py {gpu_version} first to train the model.")
        print(f"Usage: python evaluate_svm.py [naive|shared|v3]")
        return
    
    if not os.path.exists(test_features_path):
        print(f"ERROR: Test features not found at {test_features_path}")
        folder_map = {'naive': 'GPU_naive', 'shared': 'GPU_v2_shared_mem', 'v3': 'GPU_v3_optimized_gradient'}
        print(f"Please run the feature extraction in {folder_map.get(gpu_version, 'GPU')} folder first.")
        print(f"Usage: python evaluate_svm.py [naive|shared|v3]")
        return
    
    if not os.path.exists(test_labels_path):
        print(f"ERROR: Test labels not found at {test_labels_path}")
        return
    
    # Load model
    svm_model = load_model(model_path)
    
    # Load test data
    test_features, test_labels = load_features(test_features_path, test_labels_path)
    
    # Evaluate model
    predictions, accuracy, conf_matrix = evaluate_svm(svm_model, test_features, test_labels)
    
    # Print results
    print_results(test_labels, predictions, accuracy, conf_matrix)
    
    # Save results
    save_results(accuracy, conf_matrix, predictions, results_path)
    
    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix, confusion_matrix_path)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
