# Runtime probes for 03-generic-address-space
# Target: Jetson Orin (sm_87) by default
# Override: make SM=sm_90

SM ?= sm_87
NVCC ?= /usr/local/cuda/bin/nvcc

SUBDIRS = 01-local-memory

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ SM=$(SM) NVCC=$(NVCC)

clean:
	for d in $(SUBDIRS); do $(MAKE) -C $$d clean; done
