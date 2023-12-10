#include "adam.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

struct Optimizer *OptmizerADAM() {
    struct ADAMOptimizer *optimizer = (struct ADAMOptimizer*)malloc(sizeof(ADAMOptimizer));
    if (optimizer == NULL) {
        printf("ADAM Optimizer malloc failed!\n");
        exit(0);
    }

    return &optimizer->base;
}