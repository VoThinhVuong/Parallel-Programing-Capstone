# Makefile for Automating cuML Installation
# USAGE:
#   make install-conda  -> Installs via Conda (Best for stability)
#   make install-pip    -> Installs via Pip (Best for existing envs)
#   make verify         -> Checks if installation was successful
#   make clean          -> Removes the conda environment

# Configuration
ENV_NAME = cuml_env
PYTHON_VER = 3.10
CUDA_VER = 12
RAPIDS_VER = 24.12

# Detected Shell (needed for conda activation)
SHELL := /bin/bash
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

.PHONY: help install-conda install-pip verify clean check-gpu

help:
	@echo "================= cuML Installer ================="
	@echo "Available commands:"
	@echo "  make install-conda   : Create a new Conda env and install cuML (Recommended)"
	@echo "  make install-pip     : Install cuML via Pip into current active environment"
	@echo "  make verify          : Run a quick python script to verify installation"
	@echo "  make clean           : Remove the '$(ENV_NAME)' conda environment"
	@echo "=================================================="

check-gpu:
	@echo "Checking for NVIDIA GPU..."
	@nvidia-smi || (echo "Error: NVIDIA GPU not found or drivers missing."; exit 1)
	@echo "GPU Check passed."

install-conda: check-gpu
	@echo "Creating Conda environment '$(ENV_NAME)'..."
	conda create -n $(ENV_NAME) -c rapidsai -c conda-forge -c nvidia \
		cuml=$(RAPIDS_VER) \
		python=$(PYTHON_VER) \
		cuda-version=$(CUDA_VER) \
		-y
	@echo "--------------------------------------------------"
	@echo "Installation Complete!"
	@echo "To activate your environment, run:"
	@echo "  conda activate $(ENV_NAME)"
	@echo "Then run 'make verify'"

install-pip: check-gpu
	@echo "Installing via PIP (Experimental)..."
	@echo "Ensure you are in a virtual environment before proceeding."
	pip install \
		--extra-index-url=https://pypi.nvidia.com \
		cuml-cu$(CUDA_VER)==$(RAPIDS_VER)
	@echo "Installation Complete."

verify:
	@echo "Running verification script..."
	@python -c "import cuml; import cuml.genomics; print(f'Success! Installed cuML version: {cuml.__version__}')" || \
	(echo "Verification FAILED. Did you activate the environment?"; exit 1)

clean:
	@echo "Removing Conda environment '$(ENV_NAME)'..."
	conda env remove -n $(ENV_NAME)