#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../../memory/mem.h"
#include "../ops.h"
#include "../compute_node.h"

struct NamedTensor *op_pow_forword(struct ComputeNode *node) {
    ComputeNode *left = node->operator.binaryOperand.left;
    ComputeNode *right = node->operator.binaryOperand.right;
    NamedTensor *leftVal = forword(left);
    NamedTensor *rightVal = forword(right);
    if (leftVal->dimension_nums == 0 && rightVal->dimension_nums == 0) {
        // FIXME: memory leak below
        return Scalar(pow(*leftVal->data, *rightVal->data));
    }
    printf("TODO: not support vector and maxrix now!\n");
    return NULL;
}

struct NamedTensor *op_pow_backword(struct ComputeNode *node) {
    ComputeNode *left = node->operator.binaryOperand.left;
    ComputeNode *right = node->operator.binaryOperand.right;
    NamedTensor *leftVal = forword(left);
    NamedTensor *rightVal = forword(right);
    if (leftVal->dimension_nums == 0 && rightVal->dimension_nums == 0) {
        double gradVal = *rightVal->data * *leftVal->data * *left->grad->data;
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

OperatorFunc op_pow = {
    .type = POW,
    .forword = op_pow_forword,
    .backward = op_pow_backword,
};

ComputeNode *Pow(ComputeNode *left, ComputeNode *right) {
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
    node->operator.op = &op_pow;

    if (left == NULL || right == NULL) {
        printf("The operand of pow should not not be NULL!\n");
        exit(1);
    }
    left->parent = node;
    right->parent = node;
    return node;
}