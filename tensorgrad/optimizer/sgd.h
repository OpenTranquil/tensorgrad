#ifndef __OPTIMIZER_SGD_H__
#define __OPTIMIZER_SGD_H__

#include "optimizer.h"

typedef struct SGDOptimizer {
    Optimizer base;
    double learningRate;
} SGDOptimizer;

struct Optimizer *OptmizerSGD(double learningRate);

#endif /* __OPTIMIZER_SGD_H__ */