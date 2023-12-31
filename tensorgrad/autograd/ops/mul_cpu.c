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
        printf("cannot mul a row vector with a row vector!\n");
        exit(0);
    }
    if (leftVal->type == TENSOR_TYPE_COLUMN_VECTOR && rightVal->type == TENSOR_TYPE_COLUMN_VECTOR) {
        printf("cannot mul a column vector with a column vector!\n");
        exit(0);
    }
    //           [ q w e ]
    // [b n m] * [ a s d ] = [bq+na+mz bw+ns+mx be+nd+mc]
    //           [ z x c ]
    if (leftVal->type == TENSOR_TYPE_ROW_VECTOR && rightVal->type == TENSOR_TYPE_MATRIX) {
        DimensionDef *leftHeight = leftVal->dimensions;
        DimensionDef *leftWidth = ContainerOf(leftHeight->node.next, DimensionDef, node);
        DimensionDef *rightHeight = rightVal->dimensions;
        DimensionDef *rightWidth = ContainerOf(rightHeight->node.next, DimensionDef, node);
        if (leftWidth->size != rightHeight->size) {
            printf("Row Vector (%d) can not mul Matrix(%dx%d)!\n", leftWidth->size, rightHeight->size, rightWidth->size);
            exit(0);
        }
        double *outData = AllocMem(rightHeight->size * sizeof(double));
        for (size_t i = 0; i < rightWidth->size; i++) {
            double val = 0.0f;
            for (size_t j = 0; j < rightHeight->size; j++) {
                val += leftVal->data[j] * rightVal->data[j * rightWidth->size + i];
            }
            outData[i] = val;
        }

        NamedTensor *output = RowVector(Dimension("mul_out", rightHeight->size), outData);
        return output;
    }
    if (leftVal->type == TENSOR_TYPE_MATRIX && rightVal->type == TENSOR_TYPE_ROW_VECTOR) {
        printf("cannot mul a matrix with a row vector!\n");
        exit(0);
    }
    //  [ q w e ]   [b]          [bq + nw + em]
    //  [ a s d ] * [n]    =     [ba + ns + md]
    //  [ z x c ]   [m]          [bz + xn + mc]
    //  
    if (leftVal->type == TENSOR_TYPE_MATRIX && rightVal->type == TENSOR_TYPE_COLUMN_VECTOR) {
        DimensionDef *leftHeight = leftVal->dimensions;
        DimensionDef *leftWidth = ContainerOf(leftHeight->node.next, DimensionDef, node);
        DimensionDef *rightHeight = rightVal->dimensions;
        DimensionDef *rightWidth = ContainerOf(rightHeight->node.next, DimensionDef, node);
        if (leftWidth->size != rightHeight->size) {
            printf("Matrix (%dx%d) can not mul Column Vector(%d)!\n", leftHeight->size, leftWidth->size, rightHeight->size);
            exit(0);
        }
        double *outData = AllocMem(rightHeight->size * sizeof(double));
        for (size_t i = 0; i < leftHeight->size; i++) {
            double val = 0.0f;
            for (size_t j = 0; j < leftWidth->size; j++) {
                val += rightVal->data[j] * leftVal->data[i * leftWidth->size + j];
            }
            outData[i] = val;
        }

        NamedTensor *output = ColumnVector(Dimension("mul_out", rightHeight->size), outData);
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

    //F(X) = X * A
    //F'(X) = At outproduct I
    if (leftVal->type == TENSOR_TYPE_MATRIX && rightVal->type == TENSOR_TYPE_COLUMN_VECTOR) {
        if (left->requireGrad) {

        }
    }

    printf("TODO: not support mul backword for %s and %s now!\n", TensorTypeName(leftVal->type), TensorTypeName(rightVal->type));
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