# Makefile for CIFAR-10 CNN Training (CPU Baseline)

# Detect OS
ifeq ($(OS),Windows_NT)
    EXECUTABLE = cifar10_cnn.exe
    RM = del /Q
    RMDIR = rmdir /S /Q
else
    EXECUTABLE = cifar10_cnn
    RM = rm -f
    RMDIR = rm -rf
endif

CC = gcc
CFLAGS = -O3 -Wall -Wextra -std=c99
LDFLAGS = -lm

# Source files
SOURCES = main.c data_loader.c cnn.c forward.c backward.c
OBJECTS = $(SOURCES:.c=.o)

# Default target
all: $(EXECUTABLE)

# Link
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)
	@echo "Build complete: $(EXECUTABLE)"

# Compile
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean
clean:
	$(RM) $(OBJECTS) $(EXECUTABLE)
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
