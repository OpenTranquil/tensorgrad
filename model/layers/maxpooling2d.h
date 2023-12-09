#ifndef __LAYER_MAXPOOLING2D_H__
#define __LAYER_MAXPOOLING2D_H__

#include "../layer.h"

typedef struct MaxPooling2DLayer {
    struct Layer base;
    TupleU64 *kernelSize;
} MaxPooling2DLayer;

struct Layer *MaxPooling2D(TupleU64 *kernel_size);

#endif /* __LAYER_MAXPOOLING2D_H__ */