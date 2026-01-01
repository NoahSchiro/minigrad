NVCC = nvcc
TARGET = main
SRC_C = main.c
SRC_CU = vector.cu
OBJ = main.o vector.o

#CUDA_ARCH = -arch=sm_86

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(OBJ) -o $(TARGET)

main.o: $(SRC_C)
	$(NVCC) -c $(SRC_C) -o main.o

vector.o: $(SRC_CU)
	$(NVCC) -c $(SRC_CU) -o vector.o

clean:
	rm -f $(TARGET) *.o
