#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../../memory/mem.h"
#include "../ops.h"
#include "../compute_node.h"

struct NamedTensor *op_softmax_forword(struct ComputeNode *node) {
    ComputeNode *operand = node->operator.unaryOperand;
    NamedTensor *operandVal = forword(operand);

    DimensionDef *operandHeight = operandVal->dimensions;
    DimensionDef *operandWidth = ContainerOf(operandHeight->node.next, DimensionDef, node);

    double expSum = 0.0f;
    double max = 0.0f;
    double *data = NULL;
    if (operandVal->type == TENSOR_TYPE_COLUMN_VECTOR) {
        for (size_t i = 0; i < operandHeight->size; i++) {
            if (operandVal->data[i] > max) {
                max = operandVal->data[i];
            }
        }
        data = AllocMem(sizeof(double) * operandHeight->size);
        for (size_t i = 0; i < operandHeight->size; i++) {
            data[i] = exp(operandVal->data[i] - max);
            expSum += data[i];
        }

        for (size_t i = 0; i < operandHeight->size; i++) {
            data[i] = data[i] / expSum;
        }
    } else if (operandVal->type == TENSOR_TYPE_ROW_VECTOR) {
        for (size_t i = 0; i < operandWidth->size; i++) {
            if (operandVal->data[i] > max) {
                max = operandVal->data[i];
            }
        }
        data = AllocMem(sizeof(double) * operandWidth->size);
        for (size_t i = 0; i < operandWidth->size; i++) {
            data[i] = exp(operandVal->data[i] - max);
            expSum += data[i];
        }

        for (size_t i = 0; i < operandWidth->size; i++) {
            data[i] = data[i] / expSum;
        }
    }

    return RowVector(Dimension("softmax_out", 10), data);
}

struct NamedTensor *op_softmax_backword(struct ComputeNode *node) {
    ComputeNode *operand = node->operator.unaryOperand;
    // TODO
    return node->grad;
}

OperatorFunc op_softmax = {
    .type = SOFTMAX,
    .forword = op_softmax_forword,
    .backward = op_softmax_backword,
};

ComputeNode *Softmax(ComputeNode *operand) {
    ComputeNode *node = (ComputeNode *)AllocMem(sizeof(ComputeNode));
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