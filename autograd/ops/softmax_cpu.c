#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../ops.h"
#include "../compute_node.h"

struct NamedTensor *op_softmax_forword(struct ComputeNode *node) {
    ComputeNode *left = node->operator.unaryOperand;
    NamedTensor *leftVal = forword(left);

    double expSum = 0.0f;
    double max = 0.0f;
    for (size_t i = 0; i < leftVal->dimensions->size; i++) {
        if (leftVal->data[i] > max) {
            max = leftVal->data[i];
        }
    }

    double *data = malloc(sizeof(double) * leftVal->dimensions->size);
    for (size_t i = 0; i < leftVal->dimensions->size; i++) {
        data[i] = exp(leftVal->data[i] - max);
        expSum += data[i];
    }

    for (size_t i = 0; i < leftVal->dimensions->size; i++) {
        data[i] = data[i] / expSum;
    }

    NamedTensor *result = Vector(Dimension("x", 10), data);
    return result;
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