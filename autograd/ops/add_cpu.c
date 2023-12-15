#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include "../ops.h"
#include "../compute_node.h"

struct NamedTensor *op_add_forword(struct ComputeNode *node) {
    ComputeNode *left = node->operator.binaryOperand.left;
    ComputeNode *right = node->operator.binaryOperand.right;
    NamedTensor *leftVal = forword(left);
    NamedTensor *rightVal = forword(right);
    if (leftVal->dimension_nums == 0 && rightVal->dimension_nums == 0) {
        return  Scalar(*leftVal->data + *rightVal->data);
    }
    printf("TODO: not support vector and maxrix now!\n");
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
    ComputeNode *node = (ComputeNode *)malloc(sizeof(ComputeNode));
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