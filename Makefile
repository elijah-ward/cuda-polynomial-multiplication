GCC = g++

GCCFLAGS = -c

NVCC = nvcc

SRCCC =

SRCCU = poly_mult.cu

### NVCCFLAGS = -c -O2 --compiler-bindir /usr/bin//gcc-4.8
NVCCFLAGS = -c -O2 --compiler-bindir /usr/bin/

EXE = poly_mult

RM = rm -f

OBJ = $(SRCCC:.c=.o) $(SRCCU:.cu=.o)

all: $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $*.cu

clean:
	$(RM) *.o *~ *.linkinfo a.out *.log $(EXE)

test_q1:
	./$(EXE) test 1 64 103
	./$(EXE) test 1 128 103
	./$(EXE) test 1 512 103

test_q2:
	./$(EXE) test 2 64 103
	./$(EXE) test 2 128 103
	./$(EXE) test 2 512 103

test_all:
	make test_q1
	make test_q2