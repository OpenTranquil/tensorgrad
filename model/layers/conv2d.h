#ifndef __LAYER_CONV2D_H__
#define __LAYER_CONV2D_H__

#include "../layer.h"

typedef struct Conv2DLayer {
    struct Layer base;
    uint64_t filters;
    TupleU64 *kernelSize;
} Conv2DLayer;

struct Layer *Conv2D(uint64_t filters, TupleU64 *kernel_size, ActivationType actv);

#endif /* __LAYER_CONV2D_H__ */