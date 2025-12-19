# Makefile for CIFAR-10 CNN Training (CPU Baseline)

# Detect OS
ifeq ($(OS),Windows_NT)
    EXECUTABLE = cifar10_cnn.exe
    FEATURE_EXTRACTOR = extract_features.exe
    RM = del /Q
    RMDIR = rmdir /S /Q
else
    EXECUTABLE = cifar10_cnn
    FEATURE_EXTRACTOR = extract_features
    RM = rm -f
    RMDIR = rm -rf
endif

CC = gcc
CFLAGS = -O3 -Wall -Wextra -std=c99
LDFLAGS = -lm

# Source files for main program
SOURCES = main.c data_loader.c cnn.c forward.c backward.c
OBJECTS = $(SOURCES:.c=.o)

# Source files for feature extractor
FEATURE_SOURCES = extract_features.c feature_extractor.c data_loader.c cnn.c forward.c
FEATURE_OBJECTS = $(FEATURE_SOURCES:.c=.o)

# Default target
all: $(EXECUTABLE) $(FEATURE_EXTRACTOR)

# Link main program
$(EXECUTABLE): main.o data_loader.o cnn.o forward.o backward.o feature_extractor.o
	$(CC) main.o data_loader.o cnn.o forward.o backward.o feature_extractor.o -o $@ $(LDFLAGS)
	@echo "Build complete: $(EXECUTABLE)"

# Link feature extractor
$(FEATURE_EXTRACTOR): extract_features.o feature_extractor.o data_loader.o cnn.o forward.o
	$(CC) extract_features.o feature_extractor.o data_loader.o cnn.o forward.o -o $@ $(LDFLAGS)
	@echo "Build complete: $(FEATURE_EXTRACTOR)"

# Compile
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean
clean:
	$(RM) *.o $(EXECUTABLE) $(FEATURE_EXTRACTOR)
	@echo "Clean complete"

# Run main program
run: $(EXECUTABLE)
	./$(EXECUTABLE)

# Run feature extractor with default settings
extract: $(FEATURE_EXTRACTOR)
	./$(FEATURE_EXTRACTOR)

# Run feature extractor with custom batch count (usage: make extract-batches BATCHES=3)
extract-batches: $(FEATURE_EXTRACTOR)
	./$(FEATURE_EXTRACTOR) -b $(BATCHES)

# Help
help:
	@echo "Available targets:"
	@echo "  all              - Build both main program and feature extractor (default)"
	@echo "  $(EXECUTABLE)    - Build the main training program"
	@echo "  $(FEATURE_EXTRACTOR) - Build the feature extraction program"
	@echo "  clean            - Remove object files and executables"
	@echo "  run              - Build and run the main program"
	@echo "  extract          - Build and run feature extractor with default settings"
	@echo "  extract-batches  - Build and run feature extractor with custom batch count"
	@echo "                     Usage: make extract-batches BATCHES=3"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Feature extractor options:"
	@echo "  Run './$(FEATURE_EXTRACTOR) --help' for all available options"

.PHONY: all clean run extract extract-batches help
