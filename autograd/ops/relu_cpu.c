#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include "../../memory/mem.h"
#include "../ops.h"
#include "../compute_node.h"

struct NamedTensor *op_relu_forword(struct ComputeNode *node) {
    ComputeNode *operand = node->operator.unaryOperand;
    NamedTensor *operandVal = forword(operand);
    if (operandVal->dimension_nums == 1) {
        double *outData = AllocMem(sizeof(double) * operandVal->dimensions->size);
        for (size_t i = 0; i < operandVal->dimensions->size; i++) {
            outData[i] = operandVal->data[i] > 0.0f ? operandVal->data[i] : 0.0f;
        }
        NamedTensor *output = Vector(Dimension("relu_out", operandVal->dimensions->size), outData);
        return output;
    }
    if (operandVal->dimension_nums == 2) {
        // TODO
    }
    return NULL;
}

struct NamedTensor *op_relu_backword(struct ComputeNode *node) {
    ComputeNode *left = node->operator.unaryOperand;
    node->grad = left->grad;
    return node->grad;
}

OperatorFunc op_relu = {
    .type = RELU,
    .forword = op_relu_forword,
    .backward = op_relu_backword,
};

ComputeNode *ReLU(ComputeNode *operand) {
    ComputeNode *node = (ComputeNode *)AllocMem(sizeof(ComputeNode));
    if (node == NULL) {
        printf("ComputeNode malloc failed!\n");
        exit(1);
    }
    node->type = UNARY_OPERATOR;
    node->grad = Scalar(1.0f);
    node->parent == NULL;

    node->operator.unaryOperand = operand;
    node->operator.op = &op_relu;

    if (operand == NULL) {
        printf("The operand of relu should not not be NULL!\n");
        exit(1);
    }
    operand->parent = node;
    return node;
}