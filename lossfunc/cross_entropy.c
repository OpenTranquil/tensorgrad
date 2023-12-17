#include "cross_entropy.h"
#include "../common/dlist.h"
#include "../memory/mem.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double crossentropy_forword(struct LossFunc *func, struct NamedTensor *input, struct NamedTensor *expected) {
    printf("crossentropy_forword\n");
    struct LossFuncCrossEntropy *lossfunc = ContainerOf(func, LossFuncCrossEntropy, base);
    if (input == NULL || expected == NULL) {
        printf("input vector and expected vector should not be NULL!\n");
        exit(0);
    }
    if (input->dimension_nums != 1 || expected->dimension_nums != 1) {
        printf("cross entropy dimension should be 1!\n");
        exit(0);
    }
    if (input->dimensions->size != expected->dimensions->size) {
        printf("size of vector not equal!\n");
        exit(0);
    }
    input->print(input);
    expected->print(expected);
    double loss = 0.0f;
    for (size_t i = 0; i < input->dimensions->size; i++) {
        loss += (input->data[i] * log2(expected->data[i]));
    }

    return -loss;
}

struct LossFunc *CrossEntropyLossFunc() {
    struct LossFuncCrossEntropy *loss = (struct LossFuncCrossEntropy *)AallocMem(sizeof(struct LossFuncCrossEntropy));
    if (loss == NULL) {
        printf("Cross Entropy loss func malloc failed!\n");
    }

    loss->base.ops.forword = crossentropy_forword;

    return &loss->base;
}