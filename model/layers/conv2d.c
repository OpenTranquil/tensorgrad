#include "conv2d.h"
#include "../../common/random.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

void conv2d_forword(struct Layer *layer, struct Tensor *input) {
    printf("conv2d_forword\n");
    struct Conv2DLayer *conv2dLayer = ContainerOf(layer, Conv2DLayer, base);

    // TODO
}

void conv2d_backword(struct Layer *layer) {
    printf("conv2d_backword\n");
    struct Conv2DLayer *conv2dLayer = ContainerOf(layer, Conv2DLayer, base);
    // TODO
}

static conv2d_unit_create(uint64_t width, uint64_t height) {
    struct Conv2DUnit *unit = (struct Conv2DUnit*)malloc(sizeof(Conv2DUnit));
    if (unit == NULL) {
        printf("conv2d unit malloc failed!\n");
        exit(0);
    }
    dlist_init(&unit->node);
    unit->kernel.width = width;
    unit->kernel.height = height;

    float *kernel_data = (float *)malloc(sizeof(float));
    if (kernel_data == NULL) {
        printf("conv2d kernel data malloc failed!\n");
        exit(0);
    }
    for (size_t x = 0; x < unit->kernel.height; x++) {
        for (size_t y = 0; y < unit->kernel.width; y++) {
            unit->kernel.data[x * unit->kernel.height + y] = frand(1.0f);
        }
    }

    unit->kernel.data = kernel_data;
    return unit;
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

    for (size_t filterNum = 0; filterNum < filters; filterNum++) {
        struct Conv2DUnit *unit = conv2d_unit_create(kernel_size->x, kernel_size->y);
        if (layer->units == NULL) {
            layer->units == unit;
        } else {
            dlist_append_tail(&layer->units->node, &unit->node);
        }
    }

    layer->base.ops.backword = conv2d_backword;
    layer->base.ops.forword = conv2d_forword;

    return &layer->base;
}