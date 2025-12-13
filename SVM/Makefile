# Makefile for setting up and building ThunderSVM
# Works on both Linux and Windows (with make installed)

# Detect OS
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    SHARED_LIB := thundersvm.dll
    BUILD_LIB_PATH := thundersvm/build/bin/Debug/$(SHARED_LIB)
    PYTHON := python
    RM := del /Q
    RMDIR := rmdir /S /Q
else
    DETECTED_OS := $(shell uname -s)
    ifeq ($(DETECTED_OS),Linux)
        SHARED_LIB := libthundersvm.so
        BUILD_LIB_PATH := thundersvm/build/lib/$(SHARED_LIB)
    else ifeq ($(DETECTED_OS),Darwin)
        SHARED_LIB := libthundersvm.dylib
        BUILD_LIB_PATH := thundersvm/build/lib/$(SHARED_LIB)
    endif
    PYTHON := python3
    RM := rm -f
    RMDIR := rm -rf
endif

.PHONY: all install build copy clean help

# Default target
all: install build copy
	@echo "===================================================================="
	@echo "✓ ThunderSVM setup complete!"
	@echo "===================================================================="
	@echo "Library copied to: $(SHARED_LIB)"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Test setup: $(PYTHON) verify_setup.py"
	@echo "  2. Train SVM:  $(PYTHON) train_svm.py v3 --subset 15000"
	@echo "===================================================================="

# Install Python dependencies
install:
	@echo "===================================================================="
	@echo "Step 1: Installing Python dependencies..."
	@echo "===================================================================="
	$(PYTHON) -m pip install -r requirements.txt
	@echo "✓ Dependencies installed"
	@echo ""

# Clone and build ThunderSVM
build:
	@echo "===================================================================="
	@echo "Step 2: Building ThunderSVM..."
	@echo "===================================================================="
	@echo "Detected OS: $(DETECTED_OS)"
	@echo "Target library: $(SHARED_LIB)"
	@echo ""
	@if [ ! -d "thundersvm" ]; then \
		echo "Cloning ThunderSVM repository..."; \
		git clone https://github.com/Xtra-Computing/thundersvm.git; \
	else \
		echo "ThunderSVM repository already exists, skipping clone..."; \
	fi
	@echo ""
	@echo "Building ThunderSVM (this may take several minutes)..."
	cd thundersvm && \
	mkdir -p build && \
	cd build && \
	cmake -DUSE_CUDA=ON .. && \
	cmake --build . --config Debug -j
	@echo "✓ ThunderSVM built successfully"
	@echo ""

# Copy shared library to SVM folder
copy:
	@echo "===================================================================="
	@echo "Step 3: Copying shared library..."
	@echo "===================================================================="
	@if [ -f "$(BUILD_LIB_PATH)" ]; then \
		cp $(BUILD_LIB_PATH) $(SHARED_LIB); \
		echo "✓ Copied $(BUILD_LIB_PATH) -> $(SHARED_LIB)"; \
	else \
		echo "✗ ERROR: Built library not found at $(BUILD_LIB_PATH)"; \
		echo "  Build may have failed. Check error messages above."; \
		exit 1; \
	fi
	@echo ""

# Clean build artifacts
clean:
	@echo "Cleaning ThunderSVM build artifacts..."
	@if [ -d "thundersvm/build" ]; then \
		$(RMDIR) thundersvm/build; \
	fi
	@$(RM) $(SHARED_LIB) 2>/dev/null || true
	@echo "✓ Clean complete"

# Remove everything (including cloned repo)
distclean: clean
	@echo "Removing ThunderSVM repository..."
	@$(RMDIR) thundersvm 2>/dev/null || true
	@echo "✓ Full clean complete"

# Rebuild everything
rebuild: clean all

# Help target
help:
	@echo "ThunderSVM Build System"
	@echo "======================="
	@echo ""
	@echo "Targets:"
	@echo "  make all       - Install deps, build ThunderSVM, copy library (default)"
	@echo "  make install   - Install Python dependencies only"
	@echo "  make build     - Clone and build ThunderSVM only"
	@echo "  make copy      - Copy built library to SVM folder"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make distclean - Remove everything including cloned repo"
	@echo "  make rebuild   - Clean and rebuild everything"
	@echo "  make help      - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - CMake 3.11+"
	@echo "  - CUDA Toolkit (for GPU support)"
	@echo "  - Git"
	@echo "  - C++ compiler (gcc/g++ on Linux, MSVC on Windows)"
	@echo ""
	@echo "Example usage:"
	@echo "  make                    # Build everything"
	@echo "  make clean && make      # Rebuild from scratch"
