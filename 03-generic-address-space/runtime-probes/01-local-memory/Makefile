# Makefile — local memory address probe
# Inherits SM and NVCC from parent Makefile, or defaults to sm_87

SM   ?= sm_87
NVCC ?= /usr/local/cuda/bin/nvcc
TARGET = ../build/probe_local_memory

all: $(TARGET)

../build:
	mkdir -p $@

$(TARGET): probe_local_memory.cu | ../build
	$(NVCC) -arch=$(SM) -O3 -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf ../build
