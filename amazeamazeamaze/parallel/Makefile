# Compiler
NVCC = nvcc

# Compiler flags
# -O3 for optimization
NVCC_FLAGS = -O3

# Target executables
TARGETS = GPU_alltime

# Default target: build all programs
all: $(TARGETS)

# Individual build rules
GPU_alltime: GPU_alltime.cu
	$(NVCC) $(NVCC_FLAGS) GPU_alltime.cu -o GPU_alltime -lcufft

# Clean up generated files
clean:
	rm -f $(TARGETS)

