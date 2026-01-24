NVCC = nvcc
EXE = main
LIB = libcuda.so
LIB_DIR = /usr/local/lib

CFLAGS = -Xcompiler -fPIC
LDFLAGS = -shared

SRC_C = main.c
SRC_CU = cuda_utils.cu cuda_ops.cu
OBJ_CU = cuda_utils.o cuda_ops.o
OBJ = main.o ndarray_utils.o

all: $(EXE) lib

# Build shared library
lib: $(OBJ_CU) ndarray_utils.o
	$(NVCC) $(LDFLAGS) -o $(LIB) $(OBJ_CU) ndarray_utils.o

install: lib
	sudo cp $(LIB) $(LIB_DIR)
	sudo ldconfig

# Build executable
$(EXE): main.o $(OBJ_CU) ndarray_utils.o
	$(NVCC) $^ -o $@

main.o: main.c
	$(NVCC) -c $< -o $@

ndarray_utils.o: ndarray_utils.c
	$(NVCC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(EXE) $(LIB) *.o
