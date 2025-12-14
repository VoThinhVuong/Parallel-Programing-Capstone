# Makefile for Automating RAPIDS (cuML, cuDF, etc.) Installation
# USAGE:
#   make install-pip    -> Installs full RAPIDS suite via Pip
#   make verify         -> Checks if installation was successful
#   make clean-pip      -> Uninstalls the packages (use with caution)

# Configuration
# Note: RAPIDS 25.12 requires Python 3.10, 3.11, or 3.12
NV_INDEX_URL = https://pypi.nvidia.com
PIP_PACKAGES = "cudf-cu12==25.12.*" \
               "dask-cudf-cu12==25.12.*" \
               "cuml-cu12==25.12.*" \
               "cugraph-cu12==25.12.*" \
               "nx-cugraph-cu12==25.12.*" \
               "cuxfilter-cu12==25.12.*" \
               "cucim-cu12==25.12.*" \
               "pylibraft-cu12==25.12.*" \
               "raft-dask-cu12==25.12.*" \
               "cuvs-cu12==25.12.*"

.PHONY: help install-pip verify clean-pip check-gpu

help:
	@echo "================= RAPIDS Installer ================="
	@echo "Available commands:"
	@echo "  make install-pip     : Install full RAPIDS suite (25.12) via Pip"
	@echo "  make verify          : Run a quick python script to verify installation"
	@echo "===================================================="

check-gpu:
	@echo "Checking for NVIDIA GPU..."
	@nvidia-smi || (echo "Error: NVIDIA GPU not found or drivers missing."; exit 1)
	@echo "GPU Check passed."

install-pip: check-gpu
	@echo "Installing RAPIDS 25.12 suite via PIP..."
	@echo "Ensure you are in a virtual environment (venv/conda) before proceeding."
	pip install \
		--extra-index-url=$(NV_INDEX_URL) \
		$(PIP_PACKAGES)
	@echo "--------------------------------------------------"
	@echo "Installation Complete."

verify:
	@echo "Running verification script..."
	@# We set the env var inline here to prevent the pynvjitlink error
	@export NUMBA_CUDA_ENABLE_PYNVJITLINK=1 && python -c "import cuml; import cudf; print(f'Success! Installed cuML: {cuml.__version__} | cuDF: {cudf.__version__}')" || \
	(echo "Verification FAILED."; exit 1)

clean-pip:
	@echo "Uninstalling RAPIDS packages..."
	pip uninstall -y cudf-cu12 dask-cudf-cu12 cuml-cu12 cugraph-cu12 nx-cugraph-cu12 cuxfilter-cu12 cucim-cu12 pylibraft-cu12 raft-dask-cu12 cuvs-cu12

#===============================================================
# cuML SVM Training and Evaluation
#===============================================================

install-deps:
	@echo "Installing additional Python dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed."

# Train SVM with cuML (default parameters)
train:
	@echo "Training SVM with cuML (GPU-accelerated)..."
	python train_svm_cuml.py

# Train with custom parameters
train-custom:
	@echo "Training SVM with custom parameters..."
	python train_svm_cuml.py --C 10.0 --gamma auto --kernel rbf

# Train on subset for quick testing
train-test:
	@echo "Training SVM on subset (1000 samples) for testing..."
	python train_svm_cuml.py --subset 1000

# Evaluate SVM on test data
evaluate:
	@echo "Evaluating SVM on test data..."
	python evaluate_svm_cuml.py

# Evaluate on subset for quick testing
evaluate-test:
	@echo "Evaluating SVM on subset (500 samples) for testing..."
	python evaluate_svm_cuml.py --subset 500

# Full pipeline: train and evaluate
pipeline:
	@echo "Running full SVM pipeline (train + evaluate)..."
	$(MAKE) train
	$(MAKE) evaluate
	@echo "Pipeline complete!"

# Quick test pipeline with subsets
test-pipeline:
	@echo "Running quick test pipeline..."
	$(MAKE) train-test
	$(MAKE) evaluate-test
	@echo "Test pipeline complete!"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -f *.pkl *.png *.txt
	@echo "Cleanup complete."

# Show help for SVM commands
help-svm:
	@echo "================= cuML SVM Commands ================="
	@echo "Training:"
	@echo "  make train            : Train SVM with default parameters (C=10, gamma=auto, RBF kernel)"
	@echo "  make train-custom     : Train with custom parameters"
	@echo "  make train-test       : Quick training on 1000 samples"
	@echo ""
	@echo "Evaluation:"
	@echo "  make evaluate         : Evaluate trained model on test data"
	@echo "  make evaluate-test    : Quick evaluation on 500 samples"
	@echo ""
	@echo "Pipeline:"
	@echo "  make pipeline         : Full train + evaluate pipeline"
	@echo "  make test-pipeline    : Quick test pipeline with subsets"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean            : Remove generated files (models, plots, results)"
	@echo "  make install-deps     : Install additional Python dependencies"
	@echo "===================================================="