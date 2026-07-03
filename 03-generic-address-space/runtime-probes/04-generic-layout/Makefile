SM   ?= sm_87
NVCC ?= /usr/local/cuda/bin/nvcc
TARGET = ../build/probe_generic_layout

.PHONY: all clean rebuild run

all: $(TARGET)

../build:
	mkdir -p $@

$(TARGET): probe_generic_layout.cu | ../build
	$(NVCC) -arch=$(SM) -O3 -o $@ $<

rebuild: clean all

run: all
	./$(TARGET)

clean:
	rm -rf ../build
