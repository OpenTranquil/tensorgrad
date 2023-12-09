#ifndef __AUTOGRAD_COMPUTE_NODE_H__
#define __AUTOGRAD_COMPUTE_NODE_H__

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
    double (*forword)(struct ComputeNode *node);
    double (*backward)(struct ComputeNode *node);
} OperatorFunc;

typedef struct ComputeNode {
    NodeType type;
    struct ComputeNode *parent;

    double grad;
    union {
        struct {
            struct Node *left;
            struct Node *right;
            OperatorFunc *op;
        } operator;
        struct {
            double val;
            const char *name;
        } variable;
        struct {
            double val;
        } constant;
    };
} ComputeNode;

ComputeNode *Pow(ComputeNode *left, ComputeNode *right);
ComputeNode *Add(ComputeNode *left, ComputeNode *right);
ComputeNode *Mul(ComputeNode *left, ComputeNode *right);
ComputeNode *Param(double init_val, const char *name);
ComputeNode *Variable(double init_val, const char *name);
ComputeNode *Constant(double init_val);
double Forword(ComputeNode *node);
double Backword(ComputeNode *node);

#endif /* __AUTOGRAD_COMPUTE_NODE_H__ */