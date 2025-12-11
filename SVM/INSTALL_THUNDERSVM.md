# ThunderSVM Installation Guide for Windows

ThunderSVM requires building from source on Windows. Follow these steps carefully.

## Prerequisites

1. **CUDA Toolkit** (10.0 or higher)
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Verify installation: `nvcc --version`

2. **Visual Studio** (2017 or later) with C++ tools
   - Community edition is fine
   - Must include: "Desktop development with C++"

3. **CMake** (3.10 or higher)
   - Download from: https://cmake.org/download/
   - Add to PATH during installation
   - Verify: `cmake --version`

4. **Git**
   - Download from: https://git-scm.com/
   - Verify: `git --version`

## Installation Steps

### Option 1: Using Pre-built Binaries (Recommended for Windows)

ThunderSVM may have compatibility issues on Windows. The pip package doesn't include pre-built CUDA libraries.

**Alternative approach:** Use the Python interface with pre-built libraries:

```powershell
# Clone the repository
git clone --recursive https://github.com/Xtra-Computing/thundersvm.git
cd thundersvm

# Build using CMake
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

# Install Python package
cd ../python
pip install .
```

### Option 2: Build from Source (Full Control)

```powershell
# 1. Clone repository
git clone --recursive https://github.com/Xtra-Computing/thundersvm.git
cd thundersvm

# 2. Create build directory
mkdir build
cd build

# 3. Configure with CMake (adjust CUDA architecture for your GPU)
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75

# For different GPU architectures, use:
# RTX 30xx series: -DCMAKE_CUDA_ARCHITECTURES=86
# RTX 20xx series: -DCMAKE_CUDA_ARCHITECTURES=75
# GTX 10xx series: -DCMAKE_CUDA_ARCHITECTURES=61

# 4. Build
cmake --build . --config Release --parallel 8

# 5. Install Python bindings
cd ../python
python setup.py install
```

### Option 3: Quick Test Installation (Docker - Recommended)

If the above fails, use Docker with NVIDIA GPU support:

```powershell
# Pull ThunderSVM Docker image
docker pull xtracomputinggroup/thundersvm

# Run with GPU support
docker run --gpus all -it -v ${PWD}:/workspace xtracomputinggroup/thundersvm
```

## Verification

Test if ThunderSVM is working:

```python
from thundersvm import SVC
import numpy as np

# Create dummy data
X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)

# Train SVM
clf = SVC(kernel='rbf')
clf.fit(X, y)

print("ThunderSVM is working!")
```

## Common Issues

### Issue 1: "FileNotFoundError: Please build the library first!"

**Cause:** The pip-installed thundersvm doesn't include compiled binaries.

**Solution:** Build from source using Option 2 above.

### Issue 2: CUDA out of memory

**Solution:** Reduce training data size or use a smaller subset for testing.

### Issue 3: CMake can't find CUDA

**Solution:** 
```powershell
# Set CUDA path manually
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
```

### Issue 4: Build fails on Windows

**Solution:** Use WSL2 (Windows Subsystem for Linux) or Docker instead:

```bash
# In WSL2
git clone --recursive https://github.com/Xtra-Computing/thundersvm.git
cd thundersvm
./build.sh
cd python
pip install .
```

## If ThunderSVM Installation Fails

### Fallback Option: Use scikit-learn with optimizations

While not GPU-accelerated, scikit-learn can be optimized:

```python
from sklearn.svm import SVC

# Use larger cache and enable probability estimates
svm = SVC(
    kernel='rbf',
    C=10.0,
    gamma='auto',
    cache_size=2000,  # 2GB cache
    verbose=True
)
```

This will be slower but will still work for the project.

## Next Steps

Once ThunderSVM is installed:

1. Run verification:
   ```powershell
   python verify_setup.py
   ```

2. Train SVM:
   ```powershell
   python train_svm.py
   ```

3. Evaluate:
   ```powershell
   python evaluate_svm.py
   ```

## System Requirements

- **GPU:** NVIDIA GPU with CUDA support (Compute Capability 3.5+)
- **RAM:** 16GB+ recommended for 50,000 training samples
- **VRAM:** 4GB+ GPU memory
- **Disk:** 2GB for ThunderSVM + compiled binaries
