NVCC = nvcc
EXE = main
LIB = libcuda.so 
LIB_DIR = /usr/local/lib/

LIB_FLAGS = -Xcompiler -fPIC -shared

OBJ = main.o

SRC_C = main.c
SRC_CU = cuda_utils.cu cuda_ops.cu
OBJ_CU = cuda_utils.o  cuda_ops.o

all: $(EXE) lib

lib: $(SRC_CU) ndarray_utils.o
	$(NVCC) $(LIB_FLAGS) -o $(LIB) $(SRC_CU)

install: lib
	sudo cp $(LIB) $(LIB_DIR)
	sudo ldconfig

$(EXE): $(OBJ) $(OBJ_CU) ndarray_utils.o
	$(NVCC) $(OBJ) $(OBJ_CU) ndarray_utils.o -o $(EXE)

$(OBJ): $(SRC_C)
	$(NVCC) -c $(SRC_C) -o $(OBJ)

# Compile CUDA sources
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

ndarray_utils.o: ndarray_utils.c
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(EXE) $(LIB) *.o
