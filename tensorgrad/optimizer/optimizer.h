#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__
#include "../autograd/compute_node.h"

typedef void (*Update)(struct Optimizer *optimizer);
typedef ComputeNode *(*AddParam)(struct Optimizer *optimizer, struct ComputeNode* param);

typedef struct OptimizerOperations {
    Update update;
    AddParam addParam;
} OptimizerOperations;

typedef struct Optimizer {
    ComputeNode *params;
    OptimizerOperations ops;
} Optimizer;

// base impl
struct ComputeNode *optimizer_add_param(struct Optimizer *optimizer, struct ComputeNode* param);

#endif /* __OPTIMIZER_H__ */