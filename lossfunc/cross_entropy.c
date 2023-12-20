#include "cross_entropy.h"
#include "../common/dlist.h"
#include "../memory/mem.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double crossentropy_forword(struct LossFunc *func, struct NamedTensor *input, struct NamedTensor *expected) {
    struct LossFuncCrossEntropy *lossfunc = ContainerOf(func, LossFuncCrossEntropy, base);
    if (input == NULL || expected == NULL) {
        printf("input vector and expected vector should not be NULL!\n");
        exit(0);
    }
    if (input->type != TENSOR_TYPE_ROW_VECTOR || expected->type != TENSOR_TYPE_ROW_VECTOR) {
        printf("cross entropy dimension should be 1!\n");
        exit(0);
    }

    DimensionDef *leftHeight = input->dimensions;
    DimensionDef *leftWidth = ContainerOf(leftHeight->node.next, DimensionDef, node);
    DimensionDef *rightHeight = expected->dimensions;
    DimensionDef *rightWidth = ContainerOf(rightHeight->node.next, DimensionDef, node);
    if (leftWidth->size != leftWidth->size) {
        printf("size of vector not equal!\n");
        exit(0);
    }
    double loss = 0.0f;
    for (size_t i = 0; i < leftWidth->size; i++) {
        loss += (expected->data[i] * log2(input->data[i]));
    }

    return -loss;
}

struct LossFunc *CrossEntropyLossFunc() {
    struct LossFuncCrossEntropy *loss = (struct LossFuncCrossEntropy *)AllocMem(sizeof(struct LossFuncCrossEntropy));
    if (loss == NULL) {
        printf("Cross Entropy loss func malloc failed!\n");
    }

    loss->base.ops.forword = crossentropy_forword;

    return &loss->base;
}