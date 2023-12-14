#ifndef __AUTOGRAD_COMPUTE_NODE_H__
#define __AUTOGRAD_COMPUTE_NODE_H__

#include <stddef.h>
#include <stdbool.h>
#include "../tensor/tensor.h"

typedef enum NodeType {
    VARIABLE = 0,
    CONSTANT,
    BINARY_OPERATOR,
} NodeType;

typedef enum OperatorType {
    MUL = 0,
    ADD,
    DIV,
    POW,
    LOG,
    LN,
} OperatorType;

typedef struct OperatorFunc {
    OperatorType type;
    struct NamedTensor *(*forword)(struct ComputeNode *node);
    struct NamedTensor *(*backward)(struct ComputeNode *node);
} OperatorFunc;

typedef struct ComputeNode {
    NodeType type;
    struct ComputeNode *parent;

    bool requireGrad;
    NamedTensor *grad;
    union {
        struct {
            struct Node *left;
            struct Node *right;
            OperatorFunc *op;
        } operator;
        struct {
            struct NamedTensor *val;
            const char *name;
        } variable;
        struct {
            struct NamedTensor *val;
        } constant;
    };
} ComputeNode;

ComputeNode *Pow(ComputeNode *left, ComputeNode *right);
ComputeNode *Add(ComputeNode *left, ComputeNode *right);
ComputeNode *Mul(ComputeNode *left, ComputeNode *right);
ComputeNode *Param(struct NamedTensor *init_val, const char *name);
ComputeNode *Variable(struct NamedTensor *init_val, const char *name);
ComputeNode *Constant(struct NamedTensor *init_val);
struct NamedTensor *Forword(ComputeNode *node);
struct NamedTensor *Backword(ComputeNode *node);

#endif /* __AUTOGRAD_COMPUTE_NODE_H__ */