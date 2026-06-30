SM   ?= sm_87
NVCC ?= /usr/local/cuda/bin/nvcc
TARGET = ../build/probe_cta_identity

.PHONY: all clean rebuild run

all: $(TARGET)

../build:
	mkdir -p $@

$(TARGET): probe_cta_identity.cu | ../build
	$(NVCC) -arch=$(SM) -O3 -o $@ $<

rebuild: clean all

run: all
	./$(TARGET)

clean:
	rm -rf ../build
