#include "compute_node.h"
#include "stddef.h"
#include <stdio.h>
#include <stdlib.h>

struct NamedTensor *forword(ComputeNode *node) {
    if (node->type == VARIABLE || node->type == PARAM || node->type == CONSTANT) {
        return node->value.val;
    }
    return node->operator.op->forword(node);
}

struct NamedTensor *backward(ComputeNode *node) {
    ComputeNode *cur = node;
    while (cur != NULL) {
        if (cur->type == VARIABLE) {
            printf("cannot compute grad for variable!\n");
            exit(1);
        }
        if (cur->type == CONSTANT) {
            printf("cannot compute grad for constant!\n");
            exit(1);
        }
        if (cur->requireGrad) {
            if (cur->parent != NULL) {
                cur->parent->requireGrad = true;
            }
        }
        if (cur->type == BINARY_OPERATOR | cur->type == UNARY_OPERATOR) {
            cur->operator.op->backward(cur);
        }
        if (cur->parent == NULL) {
            return cur->grad;
        }
        cur = cur->parent;
    }
}