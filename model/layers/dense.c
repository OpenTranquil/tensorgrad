#include "dense.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

struct Layer *Dense(uint64_t units, ActivationType actv) {
    struct DenseLayer *layer = (struct DenseLayer*)malloc(sizeof(DenseLayer));
    if (layer == NULL) {
        printf("dense layer alloc failed!\n");
        exit(1);
    }

    return &layer->base;
}