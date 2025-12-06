# Makefile for CIFAR-10 CNN Training (GPU Naive Implementation)

# Detect OS
ifeq ($(OS),Windows_NT)
    EXECUTABLE = cifar10_cnn_gpu.exe
    RM = del /Q
    RMDIR = rmdir /S /Q
else
    EXECUTABLE = cifar10_cnn_gpu
    RM = rm -f
    RMDIR = rm -rf
endif

# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O3 -arch=sm_50 -std=c++11
CUDA_LIBS = -lcudart

# C compiler for data_loader.c
CC = gcc
CFLAGS = -O3 -Wall -std=c99

# Source files
CU_SOURCES = main.cu cnn.cu forward.cu backward.cu
C_SOURCES = data_loader.c
CU_OBJECTS = $(CU_SOURCES:.cu=.o)
C_OBJECTS = $(C_SOURCES:.c=.o)
ALL_OBJECTS = $(CU_OBJECTS) $(C_OBJECTS)

# Default target
all: $(EXECUTABLE)

# Link
$(EXECUTABLE): $(ALL_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(ALL_OBJECTS) -o $@ $(CUDA_LIBS)
	@echo "Build complete: $(EXECUTABLE)"

# Compile CUDA files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile C files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean
clean:
	$(RM) $(ALL_OBJECTS) $(EXECUTABLE)
	@echo "Clean complete"

# Run
run: $(EXECUTABLE)
	./$(EXECUTABLE)

# Help
help:
	@echo "Available targets:"
	@echo "  all      - Build the executable (default)"
	@echo "  clean    - Remove object files and executable"
	@echo "  run      - Build and run the program"
	@echo "  help     - Show this help message"

.PHONY: all clean run help
