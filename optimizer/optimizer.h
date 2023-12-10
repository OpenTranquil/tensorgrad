#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__

typedef void (*Update)(struct Optimizer *optimizer, struct Model *model);

typedef struct OptimizerOperations {
    Update update;
} OptimizerOperations;

typedef struct Optimizer {
    OptimizerOperations ops;
} Optimizer;

#endif /* __OPTIMIZER_H__ */