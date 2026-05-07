DEBUG = 

OPTIONS = -Xptxas -v
CUDA_ARCH ?= native
GPU_MATCH = src/gpu_match.cu

define compile_cu_test
	nvcc -std=c++17 $(DEBUG) $(OPTIONS) -arch=$(CUDA_ARCH) $(1) cu_test.cu -o $(2)
endef

define compile_gpu_match
	nvcc -std=c++17 $(DEBUG) $(OPTIONS) -arch=$(CUDA_ARCH) -c -I. $(1) -o $(2)
endef

define edit_config
	sed -i "/#include \"config_for_ae/c\#include \"config_for_ae/$(1)\" " src/config.h
endef

.PHONY:all
all:bin/table_vertex_ulb.exe bin/table_edge_ulb.exe bin/table_edge_lb.exe bin/fig_naive.exe bin/fig_local.exe bin/fig_local_global.exe  bin/fig_local_global_unroll.exe 

bin/%.exe:bin/%.o cu_test.cu;
	$(call compile_cu_test,$<,$@)
bin/%.o: src/gpu_match.cu src/gpu_match.cuh
	$(call edit_config,$(patsubst bin/%.o,%.h,$@))
	$(call compile_gpu_match,src/gpu_match.cu,$@)

.PHONY:clean
clean:
	rm -f bin/*.o bin/*.exe