#include "cross_entropy.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

struct LossFunc *CrossEntropyLossFunc() {
    struct LossFuncCrossEntropy *loss = (struct LossFuncCrossEntropy *)malloc(sizeof(struct LossFuncCrossEntropy));
    if (loss == NULL) {
        printf("Cross Entropy loss func malloc failed!\n");
    }

    return &loss->base;
}