#ifndef __NEURON_H__
#define __NEURON_H__
#include "../tensor/tensor.h"

typedef struct Activation {

} Activation;

typedef struct Neuron {
    NamedTensor *input;
    NamedTensor *weightParams;
    NamedTensor *offsetParams;
    NamedTensor *output;
    Activation *actv;
} Neuron;

#endif /* __NEURON_H__ */