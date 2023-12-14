#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include "../ops.h"
#include "../compute_node.h"

struct NamedTensor *op_mul_forword(struct ComputeNode *node) {
    ComputeNode *left = node->operator.left;
    ComputeNode *right = node->operator.right;
    NamedTensor *leftVal = forword(left);
    NamedTensor *rightVal = forword(right);
    if (leftVal->dimension_nums == 0 && rightVal->dimension_nums == 0) {
        return  Scalar(*leftVal->data * *rightVal->data);
    }
    printf("TODO: not support vector and maxrix now!\n");
    return NULL;
}

struct NamedTensor *op_mul_backword(struct ComputeNode *node) {
    ComputeNode *right = node->operator.right;
    ComputeNode *left = node->operator.left;
    NamedTensor *rightVal = forword(right);
    if (rightVal->dimension_nums == 0) {
        double gradVal = *rightVal->data * *left->grad->data;
        // FIXME: memory leak below
        node->grad = Scalar(gradVal);
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
    ComputeNode *node = (ComputeNode *)malloc(sizeof(ComputeNode));
    if (node == NULL) {
        printf("ComputeNode malloc failed!\n");
        exit(1);
    }
    node->type = BINARY_OPERATOR;
    node->grad = Scalar(1.0f);
    node->parent == NULL;
    node->operator.left = left;
    node->operator.right = right;
    node->operator.op = &op_mul;

    if (left == NULL || right == NULL) {
        printf("The operand of mul should not not be NULL!\n");
        exit(1);
    }
    left->parent = node;
    right->parent = node;
    return node;
}