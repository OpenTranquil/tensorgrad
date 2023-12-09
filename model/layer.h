#ifndef __LAYER_H__
#define __LAYER_H__

#include <stdint.h>
#include <stdlib.h>
#include "../common/dlist.h"

typedef struct TupleU64 {
    uint64_t x;
    uint64_t y;
} TupleU64;

static struct TupleU64 *Tuple(uint64_t x, uint64_t y) {
    return (struct TupleU64*) malloc(sizeof(struct TupleU64));
}

typedef enum {
    RELU,
    SOFTMAX,
} ActivationType;

typedef struct LayerOperations {

} LayerOperations;

typedef struct Layer {
    ActivationType activation;
    ListNode node;
    LayerOperations ops;
} Layer;

#endif /* __LAYER_H__ */