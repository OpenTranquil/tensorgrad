#ifndef __LAYER_CONV2D_H__
#define __LAYER_CONV2D_H__

#include "../layer.h"
#include "../../autograd/tensor/tensor.h"

typedef struct Conv2DKernel {
    uint64_t width;
    uint64_t height;
    float *data;
} Conv2DKernel;

typedef struct Conv2DUnit {
    ListNode node;
    struct Conv2DKernel kernel;
} Conv2DUnit;

typedef struct Conv2DLayer {
    struct Layer base;
    uint64_t filters;
    TupleU64 *kernelSize;

    Conv2DUnit *units;
} Conv2DLayer;

struct Layer *Conv2D(uint64_t filters, TupleU64 *kernel_size, ActivationType actv);


#endif /* __LAYER_CONV2D_H__ */