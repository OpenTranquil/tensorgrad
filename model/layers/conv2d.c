#include "conv2d.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

void conv2d_forword(struct Layer *layer) {
    printf("conv2d_forword\n");
    struct Conv2DLayer *conv2dLayer = ContainerOf(layer, Conv2DLayer, base);
    // TODO
}

void conv2d_backword(struct Layer *layer) {
    printf("conv2d_backword\n");
    struct Conv2DLayer *conv2dLayer = ContainerOf(layer, Conv2DLayer, base);
    // TODO
}

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

    layer->base.ops.backword = conv2d_backword;
    layer->base.ops.forword = conv2d_forword;

    return &layer->base;
}