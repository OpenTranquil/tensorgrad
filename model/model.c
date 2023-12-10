#include "model.h"
#include "../optimizer/optimizer.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

struct Layer* model_add_layer(struct NNModel *model, struct Layer *layer) {
    if (model == NULL || layer == NULL) {
        exit(1);
        printf("model or layer is NULL!\n");
    }
    if (model->layers == NULL) {
        model->layers = layer;
    } else {
        dlist_append_tail(&model->layers->node, &layer->node);
    }
    return layer;
}

struct NNModel* model_compile(struct NNModel *model, struct Optimizer *optmizer, struct LossFunc *loss) {
    model->optmizer = optmizer;
    model->lossFunc = loss;
}

struct NNModel* model_fit(struct NNModel *model, struct Tensor *data, uint64_t epochs, uint64_t batchSize, float validationSplit) {

}

struct NNModel* model_evaluate(struct NNModel *model) {

}

struct NNModel *SequentialModel() {
    struct NNModel *model = (struct NNModel*)malloc(sizeof(NNModel));
    if (model == NULL) {
        printf("model alloc failed!\n");
        exit(1);
    }

    model->layers = NULL;
    model->optmizer = NULL;

    model->addLayer = model_add_layer;
    model->compile = model_compile;
    model->fit = model_fit;
    model->evaluate = model_evaluate;

    return model;
}