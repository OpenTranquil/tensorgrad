#include "model.h"
#include "../memory/mem.h"
#include "../common/random.h"
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
        ListNode *node = &model->layers->node;
        dlist_append_tail(node, &layer->node);
    }
    return layer;
}

struct NNModel* model_compile(struct NNModel *model, struct Optimizer *optmizer, struct LossFunc *loss) {
    model->optmizer = optmizer;
    model->lossFunc = loss;
    //TODO: Contrsuct Compute Node Tree from network structure
}

struct NNModel* model_fit(struct NNModel *model, struct Tensor *tensor, uint64_t epochs, uint64_t batchSize, float validationSplit) {
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
            if (layer->node.prev == NULL) { // top layer
                layer->ops.forword(layer, tensor);
            } else {
                Layer *prevLayer = ContainerOf(layer->node.prev, Layer, node);
                // TODO:
                layer->ops.forword(layer, tensor);
            }
            node = node->next;
        }

        //TODO:
        double *data = (double *)AllocMem(sizeof(double) * 10);
        data[0] = 0.000065;
        data[1] = 0.000241;
        data[2] = 0.123577;
        data[3] = 0.006343;
        data[4] = 0.013310;
        data[5] = 0.000577;
        data[6] = 0.000103;
        data[7] = 0.057366;
        data[8] = 0.057614;
        data[9] = 0.740805;
        struct NamedTensor *lastLayerOutput = Vector(Dimension("P", 10), data);

        double *data2 = (double *)AllocMem(sizeof(double) * 10);
        for (size_t i = 0; i < 10; i++) {
            data2[i] = 0.000001f;
        }
        data2[9] = 1.0f;
        struct NamedTensor *expectedVector = Vector(Dimension("Q", 10), data2);

        LossFunc *lossfunc = model->lossFunc;
        float loss = lossfunc->ops.forword(lossfunc, lastLayerOutput, expectedVector);

        model->onLoss(model, loss);

        Optimizer *optimizer = model->optmizer;
        optimizer->ops.update(optimizer, model);
    }

    return model;
}

struct NNModel* model_evaluate(struct NNModel *model) {

}

void model_onloss(struct NNModel *model, double loss) {
    printf("LOSS: %f\n", loss);
}

struct NNModel *SequentialModel() {
    struct NNModel *model = (struct NNModel*)AllocMem(sizeof(NNModel));
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
    model->onLoss = model_onloss;

    return model;
}