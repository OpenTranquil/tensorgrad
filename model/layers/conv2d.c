#include "conv2d.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

struct Layer *Conv2D(uint64_t filters, TupleU64 *kernel_size, ActivationType actv) {
    struct Conv2DLayer *layer = (struct Conv2DLayer*)malloc(sizeof(Conv2DLayer));
    if (layer == NULL) {
        printf("conv2d layer alloc failed!\n");
        exit(1);
    }

    dlist_init(&layer->base.node);
    layer->base.activation = actv;

    layer->filters = filters;
    layer->kernelSize = kernel_size;

    return &layer->base;
}