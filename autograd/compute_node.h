#ifndef __AUTOGRAD_COMPUTE_NODE_H__
#define __AUTOGRAD_COMPUTE_NODE_H__

#include <stddef.h>
#include <stdbool.h>
#include "tensor/tensor.h"

typedef enum NodeType {
    VARIABLE = 0,
    PARAM,
    CONSTANT,
    UNARY_OPERATOR,
    BINARY_OPERATOR,
} NodeType;

typedef enum OperatorType {
    MUL = 0,
    ADD,
    DIV,
    POW,
    LOG,
    LN,
    SOFTMAX,
    RELU,
    CONV2D,
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
            union {
                struct {
                    struct Node *left;
                    struct Node *right;
                } binaryOperand;
                struct Node *unaryOperand;
            };
            OperatorFunc *op;
        } operator;
        struct {
            struct NamedTensor *val;
            const char *name;
        } value;
    };
} ComputeNode;

ComputeNode *Pow(ComputeNode *left, ComputeNode *right);
ComputeNode *Add(ComputeNode *left, ComputeNode *right);
ComputeNode *Mul(ComputeNode *left, ComputeNode *right);
ComputeNode *Softmax(ComputeNode *operand);
ComputeNode *ReLU(ComputeNode *operand);

ComputeNode *Param(struct NamedTensor *init_val, const char *name);
ComputeNode *Variable(struct NamedTensor *init_val, const char *name);
ComputeNode *Constant(struct NamedTensor *init_val);

struct NamedTensor *Forword(ComputeNode *node);
struct NamedTensor *Backword(ComputeNode *node);

#endif /* __AUTOGRAD_COMPUTE_NODE_H__ */