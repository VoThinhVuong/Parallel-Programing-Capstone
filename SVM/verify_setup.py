"""
Verify SVM setup and check feature files
"""

import os
import struct
import sys

dirname = os.path.dirname(os.path.abspath(__file__))


def check_module(module_name):
    """Check if a Python module can be imported."""
    try:
        __import__(module_name)
        print(f"✓ {module_name:20s} - Available")
        return True
    except ImportError:
        print(f"✗ {module_name:20s} - Not installed")
        return False

def check_feature_file(filepath, expected_samples):
    """Verify a feature/label binary file."""
    if not os.path.exists(filepath):
        print(f"  ✗ File not found: {filepath}")
        return False
    
    size = os.path.getsize(filepath)
    print(f"  ✓ Found: {os.path.basename(filepath)} ({size:,} bytes)")
    
    try:
        with open(filepath, 'rb') as f:
            num_samples = struct.unpack('i', f.read(4))[0]
            if num_samples == expected_samples:
                print(f"    - Samples: {num_samples} ✓")
            else:
                print(f"    - Samples: {num_samples} (expected {expected_samples}) ✗")
                return False
    except Exception as e:
        print(f"    - Error reading file: {e}")
        return False
    
    return True

def check_thundersvm_build():
    """Check if ThunderSVM shared library is built and available."""
    from sys import platform
    
#     dirname = path.dirname(path.abspath(__file__))

# if platform == "linux" or platform == "linux2":
#     shared_library_name = "libthundersvm.so"
# elif platform == "win32":
#     shared_library_name = "thundersvm.dll"
# elif platform == "darwin":
#     shared_library_name = "libthundersvm.dylib"
# else:
#     raise EnvironmentError("OS not supported!")

# if path.exists(path.abspath(path.join(dirname, shared_library_name))):
#     lib_path = path.abspath(path.join(dirname, shared_library_name))
# else:
#     if platform == "linux" or platform == "linux2":
#         lib_path = path.join(dirname, shared_library_name)
#     elif platform == "win32":
#         lib_path = path.join(dirname, shared_library_name)
#     elif platform == "darwin":
#         lib_path = path.join(dirname, shared_library_name)

# if path.exists(lib_path):
#     thundersvm = CDLL(lib_path)
# else:
#     # try the build directory
#     if platform == "linux" or platform == "linux2":
#         lib_path = path.join(dirname, "../../build/lib", shared_library_name)
#     elif platform == "win32":
#         lib_path = path.join(dirname, "../../build/lib", shared_library_name)
#     elif platform == "darwin":
#         lib_path = path.join(dirname, "../../build/lib", shared_library_name)

#     if path.exists(lib_path):
#         thundersvm = CDLL(lib_path)
#     else:
#         raise FileNotFoundError("Please build the library first!")
    
    # Implementation based on the commented logic above
    if platform == "linux" or platform == "linux2":
        shared_library_name = "libthundersvm.so"
    elif platform == "win32":
        shared_library_name = "thundersvm.dll"
    elif platform == "darwin":
        shared_library_name = "libthundersvm.dylib"
    else:
        print(f"  ✗ OS not supported: {platform}")
        return None
    
    print(f"  Looking for: {shared_library_name}")
    
    # # Try to find the library using the same logic as thundersvm
    # try:
    #     import thundersvm
    #     thundersvm_dir = os.path.dirname(os.path.abspath(thundersvm.__file__))
    # except:
    #     print(f"  ✗ ThunderSVM module not found")
    #     return None
    
    # Check in thundersvm package directory
    if os.path.exists(os.path.abspath(os.path.join(dirname, shared_library_name))):
        lib_path = os.path.abspath(os.path.join(dirname, shared_library_name))
    else:
        if platform == "linux" or platform == "linux2":
            lib_path = os.path.join(dirname, shared_library_name)
        elif platform == "win32":
            lib_path = os.path.join(dirname, shared_library_name)
        elif platform == "darwin":
            lib_path = os.path.join(dirname, shared_library_name)
    
    if os.path.exists(lib_path):
        print(f"  ✓ Found library at: {lib_path}")
        return lib_path
    else:
        # try the build directory
        if platform == "linux" or platform == "linux2":
            lib_path = os.path.join(dirname, "../../build/lib", shared_library_name)
        elif platform == "win32":
            lib_path = os.path.join(dirname, "../../build/lib", shared_library_name)
        elif platform == "darwin":
            lib_path = os.path.join(dirname, "../../build/lib", shared_library_name)
        
        if os.path.exists(lib_path):
            print(f"  ✓ Found library at: {lib_path}")
            return lib_path
        else:
            print(f"  ✗ Library not found. Expected at: {lib_path}")
            print(f"     Please build the library first!")
            return None


def main():
    print("="*70)
    print(" SVM Setup Verification")
    print("="*70)
    
    # Check Python modules
    print("\n1. Checking Python Dependencies:")
    print("-"*70)
    modules = {
        'numpy': True,
        'sklearn': True,
        'matplotlib': False,
        'seaborn': False,
        'pickle': False
    }
    
    all_required = True
    for module, required in modules.items():
        available = check_module(module)
        if required and not available:
            all_required = False
    
    # Check ThunderSVM separately (optional)
    print("\nOptional GPU acceleration:")
    thundersvm_ok = False
    lib_path = check_thundersvm_build()
    try:
        from thundersvm_lib import SVC
        print("  ✓ ThunderSVM module - Available")
        
        if lib_path:
            thundersvm_ok = True
            print("  ✓ ThunderSVM - Fully functional (GPU-accelerated)")
        else:
            print("  ⚠ ThunderSVM - Module installed but library missing")
            print("    Build instructions at: https://github.com/Xtra-Computing/thundersvm")
    except Exception as e:
        print(f"  ✗ ThunderSVM - Not available ({str(e)})")
        print(lib_path)
        print("    To install: Build from source at https://github.com/Xtra-Computing/thundersvm")
    
    if not all_required:
        print("\n⚠ Missing required dependencies!")
        print("  Run: pip install -r requirements.txt")
    
    # Check feature files
    print("\n2. Checking Feature Files:")
    print("-"*70)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    features_dir = os.path.join(parent_dir, 'extracted_features')
    
    files_ok = True
    naive_ok = True
    shared_ok = True
    
    print("\nGPU Naive version:")
    if not check_feature_file(os.path.join(features_dir, 'train_features_naive.bin'), 50000):
        naive_ok = False
    if not check_feature_file(os.path.join(features_dir, 'test_features_naive.bin'), 10000):
        naive_ok = False
    
    print("\nGPU Shared Memory version:")
    if not check_feature_file(os.path.join(features_dir, 'train_features_shared.bin'), 50000):
        shared_ok = False
    if not check_feature_file(os.path.join(features_dir, 'test_features_shared.bin'), 10000):
        shared_ok = False
    
    print("\nLabels (shared by both versions):")
    if not check_feature_file(os.path.join(features_dir, 'train_labels.bin'), 50000):
        files_ok = False
    if not check_feature_file(os.path.join(features_dir, 'test_labels.bin'), 10000):
        files_ok = False
    
    files_ok = files_ok and (naive_ok or shared_ok)
    
    if not naive_ok:
        print("\n⚠ GPU_naive feature files missing or invalid!")
        print("  Run feature extraction in GPU_naive folder:")
        print("    cd ../GPU_naive")
        print("    make extract")
    
    if not shared_ok:
        print("\n⚠ GPU_v2_shared_mem feature files missing or invalid!")
        print("  Run feature extraction in GPU_v2_shared_mem folder:")
        print("    cd ../GPU_v2_shared_mem")
        print("    make extract")
    
    # Summary
    print("\n" + "="*70)
    if all_required and files_ok:
        print(" ✓ ALL CHECKS PASSED - Ready to run SVM training!")
        print("="*70)
        print("\nNext steps:")
        if naive_ok:
            print("  GPU_naive version:")
            print("    1. Train SVM:     python train_svm.py naive")
            print("    2. Evaluate:      python evaluate_svm.py naive")
        if shared_ok:
            print("  GPU_v2_shared_mem version:")
            print("    1. Train SVM:     python train_svm.py shared")
            print("    2. Evaluate:      python evaluate_svm.py shared")
    else:
        print(" ✗ CHECKS FAILED - Please fix the issues above")
        print("="*70)
    
    return all_required and files_ok

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
