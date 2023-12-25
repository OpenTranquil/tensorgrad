#ifndef __LAYER_H__
#define __LAYER_H__

#include <stdint.h>
#include <stdlib.h>
#include "../../../tensorgrad/common/dlist.h"
#include "neuron.h"

typedef struct TupleU64 {
    uint64_t x;
    uint64_t y;
} TupleU64;

static struct TupleU64 *Tuple(uint64_t x, uint64_t y) {
    return (struct TupleU64*) malloc(sizeof(struct TupleU64));
}

typedef enum {
    ACTV_RELU,
    ACTV_SOFTMAX,
    ACTV_NONE,
} ActivationType;

typedef void (*LayerForword)(struct Layer *layer, struct Tensor *input);
typedef void (*Layerbackword)(struct Layer *layer);

typedef struct LayerOperations {
    LayerForword forword;
    Layerbackword backword;
} LayerOperations;

typedef struct Layer {
    ActivationType activation;
    Neuron *neuron;
    ListNode node;
    LayerOperations ops;
} Layer;

#endif /* __LAYER_H__ */