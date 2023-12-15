#include "cross_entropy.h"
#include "../common/dlist.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

float crossentropy_forword(struct LossFunc *func, struct NamedTensor *input, struct NamedTensor *expected) {
    printf("crossentropy_forword\n");
    struct LossFuncCrossEntropy *lossfunc = ContainerOf(func, LossFuncCrossEntropy, base);
    // TODO:
    return 0.0f;
}

struct LossFunc *CrossEntropyLossFunc() {
    struct LossFuncCrossEntropy *loss = (struct LossFuncCrossEntropy *)malloc(sizeof(struct LossFuncCrossEntropy));
    if (loss == NULL) {
        printf("Cross Entropy loss func malloc failed!\n");
    }

    loss->base.ops.forword = crossentropy_forword;

    return &loss->base;
}