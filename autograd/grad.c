#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "grad.h"

struct NamedTensor *forword(ComputeNode *node) {
    if (node->type == VARIABLE) {
        return node->variable.val;
    }
    if (node->type == CONSTANT) {
        return node->constant.val;
    }
    return node->operator.op->forword(node);
}

struct NamedTensor *backward(ComputeNode *node) {
    ComputeNode *cur = node;
    while (cur != NULL) {
        if (cur->type == VARIABLE) {
            cur->grad = Scalar(1.0f);
        }
        if (cur->type == CONSTANT) {
            printf("cannot compute grad for constant!\n");
            exit(1);
        }
        if (cur->type == BINARY_OPERATOR) {
            cur->operator.op->backward(cur);
        }
        if (cur->parent == NULL) {
            return cur->grad;
        }
        cur = cur->parent;
    }
}

struct NamedTensor *op_pow_forword(struct ComputeNode *node) {
    ComputeNode *left = node->operator.left;
    ComputeNode *right = node->operator.right;
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
    ComputeNode *left = node->operator.left;
    ComputeNode *right = node->operator.right;
    NamedTensor *leftVal = forword(left);
    NamedTensor *rightVal = forword(right);
    if (leftVal->dimension_nums == 0 && rightVal->dimension_nums == 0) {
        double gradVal = *rightVal->data * *leftVal->data * *left->grad->data;
        // FIXME: memory leak below
        node->grad = Scalar(gradVal);
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

struct NamedTensor *op_add_forword(struct ComputeNode *node) {
    ComputeNode *left = node->operator.left;
    ComputeNode *right = node->operator.right;
    NamedTensor *leftVal = forword(left);
    NamedTensor *rightVal = forword(right);
    if (leftVal->dimension_nums == 0 && rightVal->dimension_nums == 0) {
        return  Scalar(*leftVal->data + *rightVal->data);
    }
    printf("TODO: not support vector and maxrix now!\n");
    return NULL;
}

struct NamedTensor *op_add_backword(struct ComputeNode *node) {
    ComputeNode *left = node->operator.left;
    node->grad = left->grad;
    return node->grad;
}

OperatorFunc op_add = {
    .type = ADD,
    .forword = op_add_forword,
    .backward = op_add_backword,
};

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

ComputeNode *Pow(ComputeNode *left, ComputeNode *right) {
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
    node->operator.op = &op_pow;

    if (left == NULL || right == NULL) {
        printf("The operand of pow should not not be NULL!\n");
        exit(1);
    }
    left->parent = node;
    right->parent = node;
    return node;
}

ComputeNode *Add(ComputeNode *left, ComputeNode *right) {
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

    node->operator.op = &op_add;

    if (left == NULL || right == NULL) {
        printf("The operand of add should not not be NULL!\n");
        exit(1);
    }
    left->parent = node;
    right->parent = node;
    return node;
}

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

ComputeNode *Param(struct NamedTensor *init_val, const char *name) {
    ComputeNode *node = (ComputeNode *)malloc(sizeof(ComputeNode));
    if (node == NULL) {
        printf("ComputeNode malloc failed!\n");
        exit(1);
    }
    node->type = VARIABLE;
    node->grad = Scalar(1.0f);
    node->parent == NULL;
    node->requireGrad = true;

    node->variable.val = init_val;
    node->variable.name = name;
    return node;
}

ComputeNode *Variable(struct NamedTensor *init_val, const char *name) {
    ComputeNode *node = (ComputeNode *)malloc(sizeof(ComputeNode));
    if (node == NULL) {
        printf("ComputeNode malloc failed!\n");
        exit(1);
    }
    node->type = VARIABLE;
    node->grad = Scalar(1.0f);
    node->parent == NULL;
    node->requireGrad = false;

    node->variable.val = init_val;
    node->variable.name = name;
    return node;
}

ComputeNode *Constant(struct NamedTensor *init_val) {
    ComputeNode *node = (ComputeNode *)malloc(sizeof(ComputeNode));
    if (node == NULL) {
        printf("ComputeNode malloc failed!\n");
        exit(1);
    }
    node->type = CONSTANT;
    node->grad = Scalar(1.0f);
    node->parent == NULL;
    node->requireGrad = false;

    node->constant.val = init_val;
    return node;
}

struct NamedTensor *Forword(ComputeNode *node) {
    return forword(node);
}
struct NamedTensor *Backword(ComputeNode *node) {
    return backward(node);
}