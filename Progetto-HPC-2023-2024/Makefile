CC := gcc
NVCC := nvcc
CFLAGS := -std=c99 -Wall -Wpedantic
OMPFLAGS := -fopenmp
LDLIBS := -lm
MOVIEFLAGS := -DMOVIE
MOVIEDIR := movie

OMP_EXE := omp-circles
CUDA_EXE := cuda-circles

OMP_MOVIE_EXE := omp-circles-movie
CUDA_MOVIE_EXE := cuda-circles-movie

CUDA_SRC := cuda-circles.cu
OMP_SRC := omp-circles.c

all: $(OMP_EXE) $(CUDA_EXE)

cuda: $(CUDA_EXE)

omp: $(OMP_EXE)

movie: $(OMP_MOVIE_EXE) $(CUDA_MOVIE_EXE)

movie_cuda: $(CUDA_MOVIE_EXE)

movie_omp: $(OMP_MOVIE_EXE)

movie_directory:
	mkdir -p $(MOVIEDIR)

$(OMP_EXE): $(OMP_SRC)
	$(CC) -fopenmp $(CFLAGS) $< -o $@ $(LDLIBS)

$(CUDA_EXE): $(CUDA_SRC)
	$(NVCC) $< -o $@ $(LDLIBS)

$(OMP_MOVIE_EXE): $(OMP_SRC) movie_directory
	$(CC) $(MOVIEFLAGS) $(OMPFLAGS) $(CFLAGS) $< -o $@ $(LDLIBS)

$(CUDA_MOVIE_EXE): $(CUDA_SRC) movie_directory
	$(NVCC) $(MOVIEFLAGS) $< -o $@ $(LDLIBS)

.PHONY: clean

clean:
	rm -rf -d $(OMP_EXE) $(CUDA_EXE) $(OMP_MOVIE_EXE) $(CUDA_MOVIE_EXE) *.avi $(MOVIEDIR)
