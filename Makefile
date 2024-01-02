include ./make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

ARCH = sm_61

SRC = 3D.cu

EXE = 3D 

OUTPUT = *.out

LDLIBS = -lcusparse -lcublas

FLAGS = -g -G #-arch sm_20 --ptxas-options=-v
release: $(SRC)
	$(CC) $(KERNEL_DIM) $(FLAGS) -arch=$(ARCH) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) $(LDLIBS) 

enum: $(SRC)
	$(CC) $(KERNEL_DIM) $(FLAGS) -arch=$(ARCH) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) $(LDLIBS) 

debug: $(SRC)
	$(CC) $(KERNEL_DIM) $(FLAGS) -arch=$(ARCH) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) $(LDLIBS)

debugenum: $(SRC)
	$(CC) $(KERNEL_DIM) $(FLAGS) -arch=$(ARCH) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) $(LDLIBS) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo $(OUTPUT) 
