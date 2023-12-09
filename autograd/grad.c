#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "grad.h"

double forword(ComputeNode *node) {
    if (node->type == VARIABLE) {
        return node->variable.val;
    }
    if (node->type == CONSTANT) {
        return node->constant.val;
    }
    return node->operator.op->forword(node);
}

double backward(ComputeNode *node) {
    ComputeNode *cur = node;
    while (cur != NULL) {
        if (cur->type == VARIABLE) {
            cur->grad = 1.0f;
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

double op_pow_forword(struct ComputeNode *node) {
    ComputeNode *left = node->operator.left;
    ComputeNode *right = node->operator.right;
    return pow(forword(left), forword(right));
}

double op_pow_backword(struct ComputeNode *node) {
    ComputeNode *left = node->operator.left;
    ComputeNode *right = node->operator.right;
    double grad = forword(right) * forword(left) * left->grad;
    node->grad = grad;
    return grad;
}

OperatorFunc op_pow = {
    .type = POW,
    .forword = op_pow_forword,
    .backward = op_pow_backword,
};

double op_add_forword(struct ComputeNode *node) {
    ComputeNode *left = node->operator.left;
    ComputeNode *right = node->operator.right;
    return forword(left) + forword(right);
}

double op_add_backword(struct ComputeNode *node) {
    ComputeNode *left = node->operator.left;
    double grad = 1.0f * left->grad;
    node->grad = grad;
    return grad;
}

OperatorFunc op_add = {
    .type = ADD,
    .forword = op_add_forword,
    .backward = op_add_backword,
};

double op_mul_forword(struct ComputeNode *node) {
    ComputeNode *left = node->operator.left;
    ComputeNode *right = node->operator.right;

    return forword(left) * forword(right);
}

double op_mul_backword(struct ComputeNode *node) {
    ComputeNode *right = node->operator.right;
    ComputeNode *left = node->operator.left;
    double grad = forword(right) * left->grad;
    node->grad = grad;
    return grad;
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
    node->grad = 1.0f;
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
    node->grad = 1.0f;
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
    node->grad = 1.0f;
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

ComputeNode *Param(double init_val, const char *name) {
    ComputeNode *node = (ComputeNode *)malloc(sizeof(ComputeNode));
    if (node == NULL) {
        printf("ComputeNode malloc failed!\n");
        exit(1);
    }
    node->type = VARIABLE;
    node->grad = 1.0f;
    node->parent == NULL;

    node->variable.val = init_val;
    node->variable.name = name;
    return node;
}

ComputeNode *Variable(double init_val, const char *name) {
    ComputeNode *node = (ComputeNode *)malloc(sizeof(ComputeNode));
    if (node == NULL) {
        printf("ComputeNode malloc failed!\n");
        exit(1);
    }
    node->type = VARIABLE;
    node->grad = 1.0f;
    node->parent == NULL;

    node->variable.val = init_val;
    node->variable.name = name;
    return node;
}

ComputeNode *Constant(double init_val) {
    ComputeNode *node = (ComputeNode *)malloc(sizeof(ComputeNode));
    if (node == NULL) {
        printf("ComputeNode malloc failed!\n");
        exit(1);
    }
    node->type = CONSTANT;
    node->grad = 1.0f;
    node->parent == NULL;

    node->constant.val = init_val;
    return node;
}

double Forword(ComputeNode *node) {
    return forword(node);
}
double Backword(ComputeNode *node) {
    return backward(node);
}