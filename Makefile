CC = gcc

CFLAGS=-c -Wall
LDFLAGS=

OBJS_AUTOGRAD = memory/mem.o 	\
				autograd/ops.o	\
				autograd/ops/add_cpu.o	\
				autograd/ops/mul_cpu.o	\
				autograd/ops/pow_cpu.o	\
				autograd/ops/softmax_cpu.o	\
				autograd/compute_node.o	\
				tensor/tensor.o \
				lossfunc/cross_entropy.o 	\
				optimizer/sgd.o 	\
				optimizer/adam.o 	\
				model/layers/conv2d.o 	\
				model/layers/maxpooling2d.o 	\
				model/layers/flatten.o 	\
				model/layers/dense.o 	\
				model/model.o 	\
				main.o

all: grad

grad:	$(OBJS_AUTOGRAD)
	$(CC) $(LDFLAGS) $(OBJS_AUTOGRAD) -o grad

*.o: *.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf autograd/*.o
	rm -rf autograd/ops/*.o
	rm -rf grad