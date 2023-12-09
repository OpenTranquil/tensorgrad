#include "flatten.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

struct Layer *Flatten() {
    struct FlattenLayer *layer = (struct FlattenLayer*)malloc(sizeof(FlattenLayer));
    if (layer == NULL) {
        printf("flatten layer alloc failed!\n");
        exit(1);
    }

    dlist_init(&layer->base.node);

    return &layer->base;
}