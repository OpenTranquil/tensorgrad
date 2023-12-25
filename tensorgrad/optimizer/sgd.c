#include "sgd.h"
#include "../common/dlist.h"
#include "../memory/mem.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

void sgd_optimize(struct Optimizer *optimizer) {
    struct SGDOptimizer *sgdOptmizer = ContainerOf(optimizer, SGDOptimizer, base);
    // TODO:
}

struct Optimizer *OptmizerSGD(double learningRate) {
    struct SGDOptimizer *optimizer = (struct SGDOptimizer*)AllocMem(sizeof(SGDOptimizer));
    if (optimizer == NULL) {
        printf("SGD Optimizer malloc failed!\n");
        exit(0);
    }

    optimizer->base.ops.update = sgd_optimize;
    optimizer->base.ops.addParam = optimizer_add_param;
    optimizer->learningRate = learningRate;

    return &optimizer->base;
}