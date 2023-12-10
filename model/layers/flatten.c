#include "flatten.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

void flatten_forword(struct Layer *layer, struct Tensor *input) {
    printf("flatten_forword\n");
    struct FlattenLayer *flattenLayer = ContainerOf(layer, FlattenLayer, base);
    // TODO
}

void flatten_backword(struct Layer *layer) {
    printf("flatten_backword\n");
    struct FlattenLayer *flattenLayer = ContainerOf(layer, FlattenLayer, base);
    // TODO
}

struct Layer *Flatten() {
    struct FlattenLayer *layer = (struct FlattenLayer*)malloc(sizeof(FlattenLayer));
    if (layer == NULL) {
        printf("flatten layer alloc failed!\n");
        exit(1);
    }

    dlist_init(&layer->base.node);

    layer->base.ops.backword = flatten_backword;
    layer->base.ops.forword = flatten_forword;

    return &layer->base;
}