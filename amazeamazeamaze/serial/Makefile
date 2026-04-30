# Compiler
NVCC = nvcc

# Compiler flags
# -O3 for optimization
NVCC_FLAGS = -O3

# Target executables
TARGETS = CPU_alltime

# Default target: build all programs
all: $(TARGETS)

# Individual build rules
CPU_alltime: CPU_alltime.cu
	$(NVCC) $(NVCC_FLAGS) CPU_alltime.cu -o CPU_alltime

# Clean up generated files
clean:
	rm -f $(TARGETS)

