#include "maxpooling2d.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

struct Layer *MaxPooling2D(TupleU64 *kernel_size) {
    struct MaxPooling2DLayer *layer = (struct MaxPooling2DLayer*)malloc(sizeof(MaxPooling2DLayer));
    if (layer == NULL) {
        printf("max polling layer alloc failed!\n");
        exit(1);
    }

    dlist_init(&layer->base.node);
    layer->kernelSize = kernel_size;

    return &layer->base;
}