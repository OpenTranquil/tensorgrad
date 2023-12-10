#include "adam.h"
#include "../common/dlist.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

void adam_optimize(struct Optimizer *optimizer, struct Model *model) {
    printf("adam_optimize\n");
    struct ADAMOptimizer *adamOptmizer = ContainerOf(optimizer, ADAMOptimizer, base);
    // TODO
}

struct Optimizer *OptmizerADAM() {
    struct ADAMOptimizer *optimizer = (struct ADAMOptimizer*)malloc(sizeof(ADAMOptimizer));
    if (optimizer == NULL) {
        printf("ADAM Optimizer malloc failed!\n");
        exit(0);
    }

    optimizer->base.ops.update = adam_optimize;

    return &optimizer->base;
}