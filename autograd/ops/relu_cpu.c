#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include "../../memory/mem.h"
#include "../ops.h"
#include "../compute_node.h"

struct NamedTensor *op_relu_forword(struct ComputeNode *node) {
    ComputeNode *operand = node->operator.unaryOperand;
    NamedTensor *operandVal = forword(operand);
    if (operandVal->type == TENSOR_TYPE_ROW_VECTOR) {
        DimensionDef *width = ContainerOf(operandVal->dimensions->node.next, DimensionDef, node);
        double *outData = AllocMem(sizeof(double) * width->size);
        for (size_t i = 0; i < width->size; i++) {
            outData[i] = operandVal->data[i] > 0.0f ? operandVal->data[i] : 0.0f;
        }
        NamedTensor *output = RowVector(Dimension("relu_out", width->size), outData);
        return output;
    }
    if (operandVal->type == TENSOR_TYPE_COLUMN_VECTOR) {
        DimensionDef *height = operandVal->dimensions;
        double *outData = AllocMem(sizeof(double) * height->size);
        for (size_t i = 0; i < height->size; i++) {
            outData[i] = operandVal->data[i] > 0.0f ? operandVal->data[i] : 0.0f;
        }
        NamedTensor *output = ColumnVector(Dimension("relu_out", height->size), outData);
        return output;
    }
    if (operandVal->type == TENSOR_TYPE_MATRIX) {
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