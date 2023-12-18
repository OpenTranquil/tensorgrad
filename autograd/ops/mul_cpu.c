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

    if (leftVal->dimension_nums == 0 && rightVal->dimension_nums == 0) {
        return  Scalar(*leftVal->data * *rightVal->data);
    }
    if (leftVal->dimension_nums == 1 && rightVal->dimension_nums == 2) {
        DimensionDef *rightDimension1 = rightVal->dimensions;
        DimensionDef *rightDimension2 = ContainerOf(rightVal->dimensions->node.next, DimensionDef, node);
        if (leftVal->dimensions->size != rightDimension2->size) {
            printf("Vector (%d) can not mul Matrix(%dx%d)!\n", leftVal->dimensions->size, rightDimension1->size, rightDimension2->size);
            exit(0);
        }
        double *outData = AllocMem(rightDimension1->size * sizeof(double));
        for (size_t i = 0; i < rightDimension1->size; i++) {
            double val = 0.0f;
            for (size_t j = 0; j < rightDimension2->size; j++) {
                val += leftVal->data[j] * rightVal->data[i * rightDimension2->size + j];
            }
            outData[i] = val;
        }
        
        NamedTensor *output = Vector(Dimension("mal_out", rightDimension1->size), outData);
        return output;
    }
    return NULL;
}

struct NamedTensor *op_mul_backword(struct ComputeNode *node) {
    ComputeNode *right = node->operator.binaryOperand.right;
    ComputeNode *left = node->operator.binaryOperand.left;
    NamedTensor *rightVal = forword(right);
    if (rightVal->dimension_nums == 0) {
        double gradVal = *rightVal->data * *left->grad->data;
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