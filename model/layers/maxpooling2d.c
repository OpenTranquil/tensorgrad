#include "maxpooling2d.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

void maxpooling2d_forword(struct Layer *layer, struct Tensor *input) {
    printf("maxpooling2d_forword\n");
    struct MaxPooling2DLayer *maxpooling2dLayer = ContainerOf(layer, MaxPooling2DLayer, base);
    // TODO
}

void maxpooling2d_backword(struct Layer *layer) {
    printf("maxpooling2d_backword\n");
    struct MaxPooling2DLayer *maxpooling2dLayer = ContainerOf(layer, MaxPooling2DLayer, base);
    // TODO
}

struct Layer *MaxPooling2D(TupleU64 *kernel_size, ActivationType actv) {
    struct MaxPooling2DLayer *layer = (struct MaxPooling2DLayer*)malloc(sizeof(MaxPooling2DLayer));
    if (layer == NULL) {
        printf("max polling layer alloc failed!\n");
        exit(1);
    }

    dlist_init(&layer->base.node);
    layer->base.activation = actv;

    layer->kernelSize = kernel_size;

    layer->base.ops.forword = maxpooling2d_forword;
    layer->base.ops.backword = maxpooling2d_backword;

    return &layer->base;
}