#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "../memory/mem.h"
#include "ops.h"
#include "compute_node.h"

ComputeNode *Param(struct NamedTensor *init_val, const char *name) {
    ComputeNode *node = (ComputeNode *)AallocMem(sizeof(ComputeNode));
    if (node == NULL) {
        printf("ComputeNode malloc failed!\n");
        exit(1);
    }
    node->type = VARIABLE;
    node->grad = Scalar(1.0f);
    node->parent == NULL;
    node->requireGrad = true;

    node->variable.val = init_val;
    node->variable.name = name;
    return node;
}

ComputeNode *Variable(struct NamedTensor *init_val, const char *name) {
    ComputeNode *node = (ComputeNode *)AallocMem(sizeof(ComputeNode));
    if (node == NULL) {
        printf("ComputeNode malloc failed!\n");
        exit(1);
    }
    node->type = VARIABLE;
    node->grad = Scalar(1.0f);
    node->parent == NULL;
    node->requireGrad = false;

    node->variable.val = init_val;
    node->variable.name = name;
    return node;
}

ComputeNode *Constant(struct NamedTensor *init_val) {
    ComputeNode *node = (ComputeNode *)AallocMem(sizeof(ComputeNode));
    if (node == NULL) {
        printf("ComputeNode malloc failed!\n");
        exit(1);
    }
    node->type = CONSTANT;
    node->grad = Scalar(1.0f);
    node->parent == NULL;
    node->requireGrad = false;

    node->constant.val = init_val;
    return node;
}

struct NamedTensor *Forword(ComputeNode *node) {
    return forword(node);
}
struct NamedTensor *Backword(ComputeNode *node) {
    return backward(node);
}