#include "sgd.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

struct Optimizer *OptmizerSGD() {
    struct SGDOptimizer *optimizer = (struct SGDOptimizer*)malloc(sizeof(SGDOptimizer));
    if (optimizer == NULL) {
        printf("SGD Optimizer malloc failed!\n");
        exit(0);
    }

    return &optimizer->base;
}