CC = gcc

CFLAGS=-c -Wall
LDFLAGS=

OBJS_TENSORGRAD = tensorgrad/memory/mem.o 	\
				tensorgrad/autograd/tensor/tensor.o \
				tensorgrad/autograd/ops.o	\
				tensorgrad/autograd/ops/add_cpu.o	\
				tensorgrad/autograd/ops/mul_cpu.o	\
				tensorgrad/autograd/ops/pow_cpu.o	\
				tensorgrad/autograd/ops/relu_cpu.o	\
				tensorgrad/autograd/ops/softmax_cpu.o	\
				tensorgrad/autograd/compute_node.o	\
				tensorgrad/lossfunc/cross_entropy.o 	\
				tensorgrad/optimizer/optimizer.o 	\
				tensorgrad/optimizer/sgd.o 	\
				tensorgrad/optimizer/adam.o

OBJS_EXAMPLE_MINNET = $(OBJS_TENSORGRAD)	\
				examples/min_net/main.o

OBJS_EXAMPLE_SCALAR_TEST = $(OBJS_TENSORGRAD)	\
				examples/scalar_grad/main.o

OBJS_EXAMPLE_MINST = $(OBJS_TENSORGRAD)	\
				examples/resnet/model/layers/conv2d.o 	\
				examples/resnet/model/layers/maxpooling2d.o 	\
				examples/resnet/model/layers/flatten.o 	\
				examples/resnet/model/layers/dense.o 	\
				examples/resnet/model/model.o 	\
				examples/resnet/main.o

all: tensorgrad

tensorgrad:	$(OBJS_TENSORGRAD)
	$(CC) $(LDFLAGS) $(OBJS_TENSORGRAD) -shared -o libtg.so
	make clean

min_net: $(OBJS_EXAMPLE_MINNET)
	$(CC) $(LDFLAGS) $(OBJS_EXAMPLE_MINNET) -o min_net
	make clean

scalar_grad: $(OBJS_EXAMPLE_SCALAR_TEST)
	$(CC) $(LDFLAGS) $(OBJS_EXAMPLE_SCALAR_TEST) -o scalar_grad_test
	make clean

minst: $(OBJS_EXAMPLE_MINST)
	$(CC) $(LDFLAGS) $(OBJS_EXAMPLE_MINST) -o minst
	make clean

*.o: *.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf *.o
	rm -rf */*.o
	rm -rf */*/*.o
	rm -rf */*/*/*.o
	rm -rf */*/*/*/*.o
	rm -rf */*/*/*/*/*.o


cleanall:
	make clean
	rm -rf minst
	rm -rf scalar_grad
	rm -rf min_net
