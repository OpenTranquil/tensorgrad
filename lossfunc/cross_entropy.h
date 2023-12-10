#ifndef __LOSS_FUNC_CROSS_ENTROPY_H__
#define __LOSS_FUNC_CROSS_ENTROPY_H__

#include "lossfunc.h"

typedef struct LossFuncCrossEntropy {
    struct LossFunc base;
} LossFuncCrossEntropy;

struct LossFunc *CrossEntropyLossFunc();

#endif /* __LOSS_FUNC_CROSS_ENTROPY_H__ */