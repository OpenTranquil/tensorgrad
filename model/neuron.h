#ifndef __NEURON_H__
#define __NEURON_H__
#include "../common/dlist.h"
#include "../tensor/tensor.h"

typedef struct Activation {

} Activation;

typedef struct Neuron {
    ListNode node;
    NamedTensor *input;
    NamedTensor *weightParams;
    NamedTensor *offsetParams;
    NamedTensor *output;
    Activation *actv;
} Neuron;

#endif /* __NEURON_H__ */