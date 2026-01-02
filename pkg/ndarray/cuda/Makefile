NVCC = nvcc
EXE = main
LIB = libcuda.so 
LIB_DIR = /usr/local/lib/

LIB_FLAGS = -Xcompiler -fPIC -shared

OBJ = main.o vector.o

SRC_C = main.c
SRC_CU = vector.cu

all: $(EXE) lib

lib: $(SRC_CU)
	$(NVCC) $(LIB_FLAGS) -o $(LIB) $(SRC_CU)

install: lib
	sudo cp $(LIB) $(LIB_DIR)
	sudo ldconfig

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE)

main.o: $(SRC_C)
	$(NVCC) -c $(SRC_C) -o main.o

vector.o: $(SRC_CU)
	$(NVCC) -c $(SRC_CU) -o vector.o

clean:
	rm -f $(EXE) $(LIB) *.o
