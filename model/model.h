#ifndef __MODEL_H__
#define __MODEL_H__

#include <stdint.h>
#include <stdlib.h>
#include "../common/dlist.h"
#include "layer.h"
#include "../optimizer/optimizer.h"
#include "../lossfunc/lossfunc.h"
#include "../tensor/tensor.h"

typedef struct Layer* (*ModelAddLayer)(struct NNModel *model, struct Layer *layer);
typedef struct NNModel* (*ModelCompile)(struct NNModel *model, struct Optimizer *optmizer, struct LossFunc *loss);
typedef struct NNModel* (*ModelTrain)(struct NNModel *model, struct Tensor *data, uint64_t epochs, uint64_t batchSize, float validationSplit);
typedef struct NNModel* (*ModelEvaluate)(struct NNModel *model);

typedef struct NNModel {
    Layer *layers;

    Optimizer *optmizer;
    LossFunc *lossFunc;

    ModelCompile compile;
    ModelTrain fit;
    ModelEvaluate evaluate;

    ModelAddLayer addLayer;
} NNModel;

struct NNModel *SequentialModel();

#endif /* __MODEL_H__ */