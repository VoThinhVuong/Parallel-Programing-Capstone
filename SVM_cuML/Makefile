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