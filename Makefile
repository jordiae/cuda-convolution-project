#CC = gcc
#OBJECTS = imfilter.c
#LIBS =
#CFLAGS = -Wall -Wextra -O2 -std=c99
#BINDIR = $(DESTDIR)/usr/bin
#NAME = imfilter
#
#imfilter: $(OBJECTS)
#	$(CC) $(CFLAGS) -o $(NAME) $(OBJECTS) $(LIBS)
#
#clean:
#	rm $(NAME)

CUDA_HOME   = /Soft/cuda/8.0.61

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -Wno-deprecated-gpu-targets -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
EXE	        = imfilter.exe
OBJ	        = imfilter.o

default: $(EXE)

imfilter.o: imfilter.cu
	$(NVCC) -c -o $@ imfilter.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)