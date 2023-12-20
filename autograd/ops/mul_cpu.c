#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include "../../memory/mem.h"
#include "../ops.h"
#include "../compute_node.h"

struct NamedTensor *op_mul_forword(struct ComputeNode *node) {
    ComputeNode *left = node->operator.binaryOperand.left;
    ComputeNode *right = node->operator.binaryOperand.right;
    NamedTensor *leftVal = forword(left);
    NamedTensor *rightVal = forword(right);

    if (leftVal->type == TENSOR_TYPE_SCALAR && rightVal->type == TENSOR_TYPE_SCALAR) {
        return  Scalar(*leftVal->data * *rightVal->data);
    }
    if (leftVal->type == TENSOR_TYPE_ROW_VECTOR && rightVal->type == TENSOR_TYPE_ROW_VECTOR) {
        // TODO
    }
    if (leftVal->type == TENSOR_TYPE_COLUMN_VECTOR && rightVal->type == TENSOR_TYPE_COLUMN_VECTOR) {
        // TODO
    }
    //           [ q w e ]   [bq + nw + em]
    // [b n m] * [ a s d ] = [ba + ns + md]
    //           [ z x c ]   [bz + xn + mc]
    if (leftVal->type == TENSOR_TYPE_ROW_VECTOR && rightVal->type == TENSOR_TYPE_MATRIX) {
        DimensionDef *leftHeight = leftVal->dimensions;
        DimensionDef *leftWidth = ContainerOf(leftHeight->node.next, DimensionDef, node);
        DimensionDef *rightHeight = rightVal->dimensions;
        DimensionDef *rightWidth = ContainerOf(rightHeight->node.next, DimensionDef, node);
        if (leftWidth->size != rightWidth->size) {
            printf("Vector (%d) can not mul Matrix(%dx%d)!\n", leftWidth->size, rightHeight->size, rightWidth->size);
            exit(0);
        }
        double *outData = AllocMem(rightHeight->size * sizeof(double));
        for (size_t i = 0; i < rightHeight->size; i++) {
            double val = 0.0f;
            for (size_t j = 0; j < rightWidth->size; j++) {
                val += leftVal->data[j] * rightVal->data[i * rightWidth->size + j];
            }
            outData[i] = val;
        }
        
        NamedTensor *output = ColumnVector(Dimension("mul_out", rightHeight->size), outData);
        return output;
    }
    //  [ q w e ]             [bq + nw + em]
    //  [ a s d ] * [b n m] = [ba + ns + md]
    //  [ z x c ]             [bz + xn + mc]
    if (leftVal->type == TENSOR_TYPE_MATRIX && rightVal->type == TENSOR_TYPE_ROW_VECTOR) {
        DimensionDef *leftHeight = leftVal->dimensions;
        DimensionDef *leftWidth = ContainerOf(leftHeight->node.next, DimensionDef, node);
        DimensionDef *rightHeight = rightVal->dimensions;
        DimensionDef *rightWidth = ContainerOf(rightHeight->node.next, DimensionDef, node);
        if (leftWidth->size != rightWidth->size) {
            printf("Matrix (%dx%d) can not mul Vector(%d)!\n", leftHeight->size, leftWidth->size, rightWidth->size);
            exit(0);
        }
        double *outData = AllocMem(leftHeight->size * sizeof(double));
        for (size_t i = 0; i < leftHeight->size; i++) {
            double val = 0.0f;
            for (size_t j = 0; j < leftWidth->size; j++) {
                val += rightVal->data[j] * leftVal->data[i * leftWidth->size + j];
            }
            outData[i] = val;
        }

        NamedTensor *output = ColumnVector(Dimension("mul_out", leftWidth->size), outData);
        return output;
    }
    if (leftVal->type == TENSOR_TYPE_MATRIX && rightVal->type == TENSOR_TYPE_MATRIX) {
        // TODO
    }
    return NULL;
}

struct NamedTensor *op_mul_backword(struct ComputeNode *node) {
    ComputeNode *right = node->operator.binaryOperand.right;
    ComputeNode *left = node->operator.binaryOperand.left;
    NamedTensor *leftVal = forword(left);
    NamedTensor *rightVal = forword(right);
    if (leftVal->type == TENSOR_TYPE_SCALAR && rightVal->type == TENSOR_TYPE_SCALAR) {
        double gradVal = 1.0f;
        if (left->requireGrad) {
            // (x*a)'=a
            gradVal = *rightVal->data * *left->grad->data;
        } else if (right->requireGrad) {
            // (a*x)'=a
            gradVal = *leftVal->data * *right->grad->data;
        } else {
            printf("No operand required gard in mul operation?!\n");
            exit(1);
        }
        // FIXME: memory leak below
        if (node->grad == NULL) {
            node->grad = Scalar(gradVal);
        } else {
            *node->grad->data = gradVal;
        }
        return node->grad;
    }


    printf("TODO: not support vector and maxrix now!\n");
    return NULL;
}

OperatorFunc op_mul = {
    .type = MUL,
    .forword = op_mul_forword,
    .backward = op_mul_backword,
};

ComputeNode *Mul(ComputeNode *left, ComputeNode *right) {
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
    node->operator.op = &op_mul;

    if (left == NULL || right == NULL) {
        printf("The operand of mul should not not be NULL!\n");
        exit(1);
    }
    left->parent = node;
    right->parent = node;
    return node;
}