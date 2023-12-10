#ifndef __OPTIMIZER_ADAM_H__
#define __OPTIMIZER_ADAM_H__

#include "optimizer.h"

typedef struct ADAMOptimizer {
    Optimizer base;
} ADAMOptimizer;

struct Optimizer *OptmizerADAM();

#endif /* __OPTIMIZER_ADAM_H__ */