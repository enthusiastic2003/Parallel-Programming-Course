CXX := g++
NVCC := nvcc

CXXFLAGS := -O3 -march=native -fopenmp -std=c++17
NVCCFLAGS :=

OPENMP_SRC := prog.cpp
CUDA_SRC := cuda_prog.cu

OPENMP_BIN := myapp
CUDA_BIN := cuda_lcs

.PHONY: all openmp cuda clean

all: $(OPENMP_BIN) $(CUDA_BIN)

$(OPENMP_BIN): $(OPENMP_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(CUDA_BIN): $(CUDA_SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

openmp: $(OPENMP_BIN)

cuda: $(CUDA_BIN)

clean:
	rm -f $(OPENMP_BIN) $(CUDA_BIN)
