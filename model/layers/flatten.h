#ifndef __LAYER_FLATTEN_H__
#define __LAYER_FLATTEN_H__

#include "../layer.h"

typedef struct FlattenLayer {
    struct Layer base;
} FlattenLayer;

struct Layer *Flatten();

#endif /* __LAYER_FLATTEN_H__ */