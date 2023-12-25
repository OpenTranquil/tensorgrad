#ifndef __LAYER_DENSE_H__
#define __LAYER_DENSE_H__

#include "../layer.h"

typedef struct DenseLayer {
    struct Layer base;
    uint64_t units;
} DenseLayer;

struct Layer *Dense(uint64_t units, ActivationType actv);

#endif /* __LAYER_DENSE_H__ */