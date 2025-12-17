# Makefile for CIFAR-10 CNN Training (GPU v4 - Further Optimized)

# Detect OS
ifeq ($(OS),Windows_NT)
    EXECUTABLE = cifar10_cnn_gpu_v4.exe
    FEATURE_EXTRACTOR = extract_features_v4.exe
    RM = del /Q
	NVCC_FLAGS = -O3 -arch=sm_50 -std=c++11
    RMDIR = rmdir /S /Q
else
    EXECUTABLE = cifar10_cnn_gpu_v4
    FEATURE_EXTRACTOR = extract_features_v4
	NVCC_FLAGS = -O3 -arch=sm_75 -std=c++11
    RM = rm -f
    RMDIR = rm -rf
endif

# CUDA compiler
NVCC = nvcc

# Compiler flags
CUDA_LIBS = -lcudart

# C compiler for data_loader.c
CC = gcc
CFLAGS = -O3 -Wall -std=c99

# Source files for main training
CU_SOURCES = main.cu cnn.cu forward.cu backward.cu decoder.cu feature_extractor.cu
C_SOURCES = data_loader.cu
CU_OBJECTS = $(CU_SOURCES:.cu=.o)
C_OBJECTS = $(C_SOURCES:.cu=.o)
ALL_OBJECTS = $(CU_OBJECTS) $(C_OBJECTS)

# Source files for feature extraction
EXTRACT_CU_SOURCES = extract_features.cu cnn.cu forward.cu feature_extractor.cu
EXTRACT_C_SOURCES = data_loader.cu
EXTRACT_CU_OBJECTS = $(EXTRACT_CU_SOURCES:.cu=_extract.o)
EXTRACT_C_OBJECTS = $(EXTRACT_C_SOURCES:.cu=_extract.o)
EXTRACT_ALL_OBJECTS = $(EXTRACT_CU_OBJECTS) $(EXTRACT_C_OBJECTS)

# Default target
all: $(EXECUTABLE) $(FEATURE_EXTRACTOR)

# Link main training program
$(EXECUTABLE): $(ALL_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(ALL_OBJECTS) -o $@ $(CUDA_LIBS)
	@echo "Build complete: $(EXECUTABLE)"

# Link feature extraction program
$(FEATURE_EXTRACTOR): $(EXTRACT_ALL_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(EXTRACT_ALL_OBJECTS) -o $@ $(CUDA_LIBS)
	@echo "Build complete: $(FEATURE_EXTRACTOR)"

# Compile CUDA files for main program
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile CUDA files for feature extractor (separate objects to avoid conflicts)
%_extract.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean
clean:
	$(RM) $(ALL_OBJECTS) $(EXTRACT_ALL_OBJECTS) $(EXECUTABLE) $(FEATURE_EXTRACTOR)
	@echo "Clean complete"

# Run main training
run: $(EXECUTABLE)
	./$(EXECUTABLE)

# Run feature extraction
extract: $(FEATURE_EXTRACTOR)
	./$(FEATURE_EXTRACTOR)

# Help
help:
	@echo "Available targets:"
	@echo "  all      - Build both executables (default)"
	@echo "  clean    - Remove object files and executables"
	@echo "  run      - Build and run the training program"
	@echo "  extract  - Build and run the feature extraction program"
	@echo "  help     - Show this help message"

.PHONY: all clean run extract help
