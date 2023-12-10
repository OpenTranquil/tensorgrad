#ifndef __OPTIMIZER_SGD_H__
#define __OPTIMIZER_SGD_H__

#include "optimizer.h"

typedef struct SGDOptimizer {
    Optimizer base;
} SGDOptimizer;

struct Optimizer *OptmizerSGD();

#endif /* __OPTIMIZER_SGD_H__ */