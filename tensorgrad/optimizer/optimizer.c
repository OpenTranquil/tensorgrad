#include <stddef.h>
#include "optimizer.h"

struct ComputeNode *optimizer_add_param(struct Optimizer *optimizer, struct ComputeNode* param) {
    if (optimizer->params == NULL) {
        optimizer->params = param;
        return optimizer->params;
    }
    dlist_append_tail(&optimizer->params->paramList, &param->paramList);
    return param;
}