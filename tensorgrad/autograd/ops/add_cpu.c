#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include "../../memory/mem.h"
#include "../ops.h"
#include "../compute_node.h"

struct NamedTensor *op_add_forword(struct ComputeNode *node) {
    ComputeNode *left = node->operator.binaryOperand.left;
    ComputeNode *right = node->operator.binaryOperand.right;
    NamedTensor *leftVal = forword(left);
    NamedTensor *rightVal = forword(right);
    if (leftVal->type == TENSOR_TYPE_SCALAR && rightVal->type == TENSOR_TYPE_SCALAR) {
        return  Scalar(*leftVal->data + *rightVal->data);
    }
    if (leftVal->type == TENSOR_TYPE_ROW_VECTOR && rightVal->type == TENSOR_TYPE_ROW_VECTOR) {
        if (leftVal->dimensions->size != rightVal->dimensions->size) { // should be 1
            printf("dimension  (%d) (%d) should equal!\n", leftVal->dimensions->size, rightVal->dimensions->size);
            exit(0);
        }
        DimensionDef *leftWidth = ContainerOf(leftVal->dimensions->node.next, DimensionDef, node);
        DimensionDef *rightWidth = ContainerOf(rightVal->dimensions->node.next, DimensionDef, node);
        if (leftWidth->size != rightWidth->size) {
            printf("dimension  (%d) (%d) should equal!\n", leftWidth->size, rightWidth->size);
            exit(0);
        }
        double *outData = AllocMem(sizeof(double) * leftWidth->size);
        for (size_t i = 0; i < leftWidth->size; i++) {
            outData[i] = leftVal->data[i] + rightVal->data[i];
        }
        NamedTensor *output = RowVector(Dimension("add_out", leftWidth->size), outData);
        return output;
    }
    if (leftVal->type == TENSOR_TYPE_COLUMN_VECTOR && rightVal->type == TENSOR_TYPE_COLUMN_VECTOR) {
        if (leftVal->dimensions->size != rightVal->dimensions->size) {
            printf("dimension (%d) (%d) should equal!\n", leftVal->dimensions->size, rightVal->dimensions->size);
            exit(0);
        }
        DimensionDef *leftWidth = ContainerOf(leftVal->dimensions->node.next, DimensionDef, node);
        DimensionDef *rightWidth = ContainerOf(rightVal->dimensions->node.next, DimensionDef, node);
        if (leftWidth->size != rightWidth->size) {
            printf("dimension (%d) (%d) should equal!\n", leftWidth->size, rightWidth->size);
            exit(0);
        }
        double *outData = AllocMem(sizeof(double) * leftVal->dimensions->size);
        for (size_t i = 0; i < leftVal->dimensions->size; i++) {
            outData[i] = leftVal->data[i] + rightVal->data[i];
        }
        NamedTensor *output = ColumnVector(Dimension("add_out", leftVal->dimensions->size), outData);
        return output;
    }
    printf("TODO: not support add backword for %s and %s now!\n", TensorTypeName(leftVal->type), TensorTypeName(rightVal->type));
    return NULL;
}

struct NamedTensor *op_add_backword(struct ComputeNode *node) {
    ComputeNode *left = node->operator.binaryOperand.left;
    node->grad = left->grad;
    return node->grad;
}

OperatorFunc op_add = {
    .type = ADD,
    .forword = op_add_forword,
    .backward = op_add_backword,
};

ComputeNode *Add(ComputeNode *left, ComputeNode *right) {
    ComputeNode *node = (ComputeNode *)AllocMem(sizeof(ComputeNode));
    if (node == NULL) {
        printf("ComputeNode malloc failed!\n");
        exit(1);
    }
    node->type = BINARY_OPERATOR;
    node->grad = Scalar(1.0f);
    node->parent == NULL;

    node->operator.binaryOperand.left = left;
    node->operator.binaryOperand.right = right;

    node->operator.op = &op_add;

    if (left == NULL || right == NULL) {
        printf("The operand of add should not not be NULL!\n");
        exit(1);
    }
    left->parent = node;
    right->parent = node;
    return node;
}