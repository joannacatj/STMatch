DEBUG =

OPTIONS = -Xptxas -v
CUDA_ARCH ?= sm_80
NVCCFLAGS = -std=c++17 $(DEBUG) $(OPTIONS) -arch=$(CUDA_ARCH) -rdc=true -I. -Ineugn -Isrc
GPU_MATCH = src/gpu_match.cu
NEUGN_OBJS = bin/neug_model.o bin/tensor_io.o bin/neugn_kernels.o bin/neugn_bridge.o

BINARIES = bin/table_vertex_ulb.exe bin/table_edge_ulb.exe bin/table_edge_lb.exe bin/table_edge_lb_find_first.exe bin/fig_naive.exe bin/fig_local.exe bin/fig_local_global.exe bin/fig_local_global_unroll.exe bin/table_edge_lb_neugn_find_first.exe

define edit_config
	sed -i "/#include \"config_for_ae/c\#include \"config_for_ae/$(1)\" " src/config.h
endef

.PHONY: all
all: $(BINARIES)

bin:
	mkdir -p bin

bin/%.exe: bin/%.o cu_test.cu $(NEUGN_OBJS) | bin
	nvcc $(NVCCFLAGS) $< cu_test.cu $(NEUGN_OBJS) -o $@

bin/%.o: src/gpu_match.cu src/gpu_match.cuh src/neugn_bridge.h src/callstack.h src/pattern.h | bin
	$(call edit_config,$(patsubst bin/%.o,%.h,$@))
	nvcc $(NVCCFLAGS) -c src/gpu_match.cu -o $@

bin/neug_model.o: neugn/neug_model.cpp neugn/neug_model.hpp neugn/tensor_io.hpp neugn/kernels.cuh | bin
	nvcc $(NVCCFLAGS) -c neugn/neug_model.cpp -o $@

bin/tensor_io.o: neugn/tensor_io.cpp neugn/tensor_io.hpp | bin
	nvcc $(NVCCFLAGS) -c neugn/tensor_io.cpp -o $@

bin/neugn_kernels.o: neugn/kernels.cu neugn/kernels.cuh | bin
	nvcc $(NVCCFLAGS) -c neugn/kernels.cu -o $@

bin/neugn_bridge.o: src/neugn_bridge.cu src/neugn_bridge.h src/callstack.h | bin
	nvcc $(NVCCFLAGS) -c src/neugn_bridge.cu -o $@

.PHONY: clean
clean:
	rm -f bin/*.o bin/*.exe
