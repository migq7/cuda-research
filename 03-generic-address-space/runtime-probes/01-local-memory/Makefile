# Makefile — local memory address probe
# Inherits SM and NVCC from parent Makefile, or defaults to sm_87

SM   ?= sm_87
NVCC ?= /usr/local/cuda/bin/nvcc
TARGET = probe_local_memory

all: $(TARGET)

$(TARGET): probe_local_memory.cu
	$(NVCC) -arch=$(SM) -O3 -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
