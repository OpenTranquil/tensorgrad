#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include "../ops.h"
#include "../compute_node.h"

struct NamedTensor *op_softmax_forword(struct ComputeNode *node) {
    ComputeNode *left = node->operator.unaryOperand;

    return NULL;
    //TODO:
}

struct NamedTensor *op_softmax_backword(struct ComputeNode *node) {
    ComputeNode *left = node->operator.unaryOperand;
    // TODO
    return node->grad;
}

OperatorFunc op_softmax = {
    .type = SOFTMAX,
    .forword = op_softmax_forword,
    .backward = op_softmax_backword,
};

ComputeNode *Softmax(ComputeNode *operand) {
    ComputeNode *node = (ComputeNode *)malloc(sizeof(ComputeNode));
    if (node == NULL) {
        printf("ComputeNode malloc failed!\n");
        exit(1);
    }
    node->type = UNARY_OPERATOR;
    node->grad = Scalar(1.0f);
    node->parent == NULL;

    node->operator.unaryOperand = operand;

    node->operator.op = &op_softmax;

    if (operand == NULL) {
        printf("The operand of softmax should not not be NULL!\n");
        exit(1);
    }
    operand->parent = node;
    return node;
}