#!/usr/bin/env python3
"""
Visualize reconstructed images from decoder and compare with original CIFAR-10 images
"""
import numpy as np
import matplotlib.pyplot as plt
import struct
import os
import argparse

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def load_reconstructed_images(filepath):
    """Load reconstructed images from binary file"""
    with open(filepath, 'rb') as f:
        # Read number of images
        num_images = struct.unpack('i', f.read(4))[0]
        # Read image size
        image_size = struct.unpack('i', f.read(4))[0]
        
        # Read all images
        data = np.fromfile(f, dtype=np.float32, count=num_images * image_size)
        
        # Reshape to (num_images, 3, 32, 32)
        images = data.reshape(num_images, 3, 32, 32)
        
        # Transpose to (num_images, 32, 32, 3) for display
        images = np.transpose(images, (0, 2, 3, 1))
        
    print(f"Loaded {num_images} reconstructed images")
    return images

def load_cifar10_test_batch(data_dir):
    """Load CIFAR-10 test batch"""
    filepath = os.path.join(data_dir, 'test_batch.bin')
    
    with open(filepath, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)
    
    # Each record is 3073 bytes (1 label + 3072 pixels)
    num_images = len(data) // 3073
    
    labels = np.zeros(num_images, dtype=np.uint8)
    images = np.zeros((num_images, 3, 32, 32), dtype=np.float32)
    
    for i in range(num_images):
        offset = i * 3073
        labels[i] = data[offset]
        img_data = data[offset + 1 : offset + 3073].astype(np.float32) / 255.0
        images[i] = img_data.reshape(3, 32, 32)
    
    # Transpose to (num_images, 32, 32, 3)
    images = np.transpose(images, (0, 2, 3, 1))
    
    return images, labels

def calculate_metrics(original, reconstructed):
    """Calculate PSNR and MSE"""
    mse = np.mean((original - reconstructed) ** 2)
    
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return mse, psnr

def plot_comparison_grid(original, reconstructed, labels, indices, save_path=None):
    """Plot a grid comparing original and reconstructed images"""
    n = len(indices)
    fig, axes = plt.subplots(3, n, figsize=(n * 2.5, 7.5))
    
    for idx, img_idx in enumerate(indices):
        orig = original[img_idx]
        recon = reconstructed[img_idx]
        label = labels[img_idx]
        
        mse, psnr = calculate_metrics(orig, recon)
        
        # Original image
        axes[0, idx].imshow(np.clip(orig, 0, 1))
        axes[0, idx].axis('off')
        if idx == 0:
            axes[0, idx].set_title(f'Original\n{CLASS_NAMES[label]}', fontsize=10)
        else:
            axes[0, idx].set_title(CLASS_NAMES[label], fontsize=10)
        
        # Reconstructed image
        axes[1, idx].imshow(np.clip(recon, 0, 1))
        axes[1, idx].axis('off')
        if idx == 0:
            axes[1, idx].set_title(f'Reconstructed\nPSNR: {psnr:.1f} dB', fontsize=10)
        else:
            axes[1, idx].set_title(f'PSNR: {psnr:.1f} dB', fontsize=10)
        
        # Difference (amplified for visibility)
        diff = np.abs(orig - recon) * 5  # Amplify difference
        axes[2, idx].imshow(np.clip(diff, 0, 1))
        axes[2, idx].axis('off')
        if idx == 0:
            axes[2, idx].set_title(f'Difference (×5)\nMSE: {mse:.4f}', fontsize=10)
        else:
            axes[2, idx].set_title(f'MSE: {mse:.4f}', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison grid to {save_path}")
    
    plt.show()

def plot_quality_distribution(original, reconstructed, save_path=None):
    """Plot distribution of reconstruction quality metrics"""
    n = len(original)
    psnrs = []
    mses = []
    
    for i in range(n):
        mse, psnr = calculate_metrics(original[i], reconstructed[i])
        psnrs.append(psnr)
        mses.append(mse)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # PSNR distribution
    axes[0].hist(psnrs, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(psnrs), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(psnrs):.2f} dB')
    axes[0].axvline(np.median(psnrs), color='green', linestyle='--', 
                    label=f'Median: {np.median(psnrs):.2f} dB')
    axes[0].set_xlabel('PSNR (dB)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Reconstruction Quality (PSNR)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MSE distribution
    axes[1].hist(mses, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(mses), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(mses):.6f}')
    axes[1].axvline(np.median(mses), color='green', linestyle='--', 
                    label=f'Median: {np.median(mses):.6f}')
    axes[1].set_xlabel('MSE')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Reconstruction Error (MSE)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved quality distribution to {save_path}")
    
    plt.show()
    
    print(f"\nReconstruction Quality Statistics:")
    print(f"  PSNR: Mean = {np.mean(psnrs):.2f} dB, Median = {np.median(psnrs):.2f} dB")
    print(f"  MSE:  Mean = {np.mean(mses):.6f}, Median = {np.median(mses):.6f}")

def plot_per_class_quality(original, reconstructed, labels, save_path=None):
    """Plot reconstruction quality per class"""
    class_psnrs = {i: [] for i in range(10)}
    
    for i in range(len(original)):
        mse, psnr = calculate_metrics(original[i], reconstructed[i])
        class_psnrs[labels[i]].append(psnr)
    
    # Calculate mean PSNR per class
    mean_psnrs = [np.mean(class_psnrs[i]) for i in range(10)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(range(10), mean_psnrs, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Class')
    ax.set_ylabel('Mean PSNR (dB)')
    ax.set_title('Reconstruction Quality by Class')
    ax.set_xticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, mean_psnrs)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved per-class quality to {save_path}")
    
    plt.show()

def interactive_viewer(original, reconstructed, labels):
    """Interactive viewer to browse through reconstructions"""
    n = len(original)
    current_idx = [0]  # Use list to allow modification in nested function
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    def update_display():
        idx = current_idx[0]
        orig = original[idx]
        recon = reconstructed[idx]
        label = labels[idx]
        
        mse, psnr = calculate_metrics(orig, recon)
        
        axes[0].clear()
        axes[0].imshow(np.clip(orig, 0, 1))
        axes[0].set_title(f'Original: {CLASS_NAMES[label]}')
        axes[0].axis('off')
        
        axes[1].clear()
        axes[1].imshow(np.clip(recon, 0, 1))
        axes[1].set_title(f'Reconstructed\nPSNR: {psnr:.2f} dB, MSE: {mse:.6f}')
        axes[1].axis('off')
        
        axes[2].clear()
        diff = np.abs(orig - recon) * 5
        axes[2].imshow(np.clip(diff, 0, 1))
        axes[2].set_title('Difference (×5)')
        axes[2].axis('off')
        
        fig.suptitle(f'Image {idx + 1}/{n} (Press left/right arrow to navigate, Q to quit)', 
                     fontsize=12)
        fig.canvas.draw()
    
    def on_key(event):
        if event.key == 'right':
            current_idx[0] = (current_idx[0] + 1) % n
            update_display()
        elif event.key == 'left':
            current_idx[0] = (current_idx[0] - 1) % n
            update_display()
        elif event.key == 'q':
            plt.close()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    update_display()
    plt.tight_layout()
    plt.show()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Visualize reconstructed CIFAR-10 images')
    parser.add_argument('--file', type=str, 
                       default='extracted_features/reconstructed_images_cpu.bin',
                       help='Path to reconstructed images binary file (default: extracted_features/reconstructed_images_cpu.bin)')
    parser.add_argument('--data-dir', type=str,
                       default='cifar-10-batches-bin',
                       help='Path to CIFAR-10 data directory (default: cifar-10-batches-bin)')
    args = parser.parse_args()
    
    # Path
    data_dir = args.data_dir
    recon_file = args.file
    print(f"Using reconstruction file: {recon_file}")
    
    # Check if files exist
    if not os.path.exists(recon_file):
        print(f"Error: Reconstructed images file not found: {recon_file}")
        print("Please run the CNN training with decoder enabled first.")
        print("\nUsage examples:")
        print("  python visualize_reconstructions.py --file extracted_features/reconstructed_images_cpu.bin")
        print("  python visualize_reconstructions.py --file extracted_features/reconstructed_images_gpu.bin")
        return
    
    print("Loading CIFAR-10 test data...")
    original, labels = load_cifar10_test_batch(data_dir)
    
    print("Loading reconstructed images...")
    reconstructed = load_reconstructed_images(recon_file)
    
    # Ensure same number of images
    n = min(len(original), len(reconstructed))
    original = original[:n]
    reconstructed = reconstructed[:n]
    labels = labels[:n]
    
    print(f"\nComparing {n} images...")
    
    # Show random sample comparison
    print("\nGenerating comparison grid...")
    sample_indices = np.random.choice(n, size=8, replace=False)
    plot_comparison_grid(original, reconstructed, labels, sample_indices, 
                        save_path='reconstruction_comparison.png')
    
    # Show quality distribution
    print("\nAnalyzing reconstruction quality...")
    plot_quality_distribution(original, reconstructed,
                             save_path='reconstruction_quality_dist.png')
    
    # Show per-class quality
    print("\nAnalyzing per-class reconstruction quality...")
    plot_per_class_quality(original, reconstructed, labels,
                          save_path='reconstruction_per_class.png')
    
    # Interactive viewer
    print("\nLaunching interactive viewer...")
    print("  Use LEFT/RIGHT arrows to navigate")
    print("  Press Q to quit")
    interactive_viewer(original, reconstructed, labels)

if __name__ == '__main__':
    main()
