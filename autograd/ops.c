#include "compute_node.h"
#include "stddef.h"
#include <stdio.h>
#include <stdlib.h>

struct NamedTensor *forword(ComputeNode *node) {
    if (node->type == VARIABLE) {
        return node->variable.val;
    }
    if (node->type == CONSTANT) {
        return node->constant.val;
    }
    return node->operator.op->forword(node);
}

struct NamedTensor *backward(ComputeNode *node) {
    ComputeNode *cur = node;
    while (cur != NULL) {
        if (cur->type == VARIABLE) {
            cur->grad = Scalar(1.0f);
        }
        if (cur->type == CONSTANT) {
            printf("cannot compute grad for constant!\n");
            exit(1);
        }
        if (cur->type == BINARY_OPERATOR) {
            cur->operator.op->backward(cur);
        }
        if (cur->parent == NULL) {
            return cur->grad;
        }
        cur = cur->parent;
    }
}