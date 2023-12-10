#include "model.h"
#include "../optimizer/optimizer.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

struct Layer* model_add_layer(struct NNModel *model, struct Layer *layer) {
    if (model == NULL || layer == NULL) {
        printf("model or layer is NULL!\n");
        exit(1);
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
    if (model == NULL) {
        printf("model is NULL!\n");
        exit(0);
    }
    if (model->layers == NULL) {
        printf("layers is NULL!\n");
        exit(0);
    }

    if (model->lossFunc == NULL) {
        printf("loss func is NULL!\n");
        exit(0);
    }

    if (model->optmizer == NULL) {
        printf("optmizer is NULL!\n");
        exit(0);
    }

    for (size_t i = 0; i < epochs; i++) {
        struct ListNode *node = &model->layers->node;
        while (node != NULL) {
            struct Layer *layer = ContainerOf(node, Layer, node);
            layer->ops.forword(layer);
            node = node->next;
        }

        LossFunc *lossfunc = model->lossFunc;
        float loss = lossfunc->ops.forword(lossfunc);

        Optimizer *optimizer = model->optmizer;
        optimizer->ops.update(optimizer, model);
    }

    return model;
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