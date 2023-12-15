#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "common/random.h"
#include "autograd/ops.h"
#include "autograd/compute_node.h"
#include "optimizer/adam.h"
#include "lossfunc/cross_entropy.h"
#include "model/layer.h"
#include "model/layers/conv2d.h"
#include "model/layers/maxpooling2d.h"
#include "model/layers/flatten.h"
#include "model/layers/dense.h"
#include "model/model.h"
#include "tensor/tensor.h"

void minist() {
    // TODO: load data
    float *data = malloc(sizeof(float) * 28 * 28);
    for (size_t i = 0; i < 28; i++) {
        for (size_t j = 0; j < 28; j++) {
            data[i * 28 + j] = frand(255.0f);
        }
    }

    NamedTensor *tensor = Tensor();
    tensor->addDimension(tensor, Dimension("height", 28));
    tensor->addDimension(tensor, Dimension("weight", 28));
    tensor->data = data;

    NNModel *model = SequentialModel();
    model->addLayer(model, Conv2D(32, Tuple(3, 3), ACTV_RELU));
    model->addLayer(model, MaxPooling2D(Tuple(2,2), ACTV_NONE));
    model->addLayer(model, Conv2D(64, Tuple(3, 3), ACTV_RELU));
    model->addLayer(model, MaxPooling2D(Tuple(2,2), ACTV_NONE));
    model->addLayer(model, Flatten());
    model->addLayer(model, Dense(64, RELU));
    model->addLayer(model, Dense(10, SOFTMAX));

    uint64_t epochs = 1;
    uint64_t batch_size = 64;
    float validation_split = 0.2;

    model->compile(model, OptmizerADAM(), CrossEntropyLossFunc());
    model->fit(model, tensor, epochs, batch_size, validation_split);
    model->evaluate(model);
}

struct NamedTensor *fx(struct NamedTensor *x, struct NamedTensor *a, struct NamedTensor *b) {
    if (x->dimension_nums == 0 && a->dimension_nums == 0 && b->dimension_nums == 0) {
        double val = pow((*a->data * *x->data + *b->data), 2);
        return Scalar(val);
    }
    // TODO
    printf("not support vector and matrix now\n");
    return NULL;
}

struct NamedTensor *fdx(struct NamedTensor *x, struct NamedTensor *a, struct NamedTensor *b) {
    if (x->dimension_nums == 0 && a->dimension_nums == 0 && b->dimension_nums == 0) {
        double grad = 2.0f * (*a->data * *x->data + *b->data) * *a->data;
        return Scalar(grad);
    }
    // TODO
    printf("not support vector and matrix now\n");
    return NULL;
}

void grad_test() {
    struct NamedTensor *av = Scalar(5.1f);
    struct NamedTensor *xv = Scalar(6.3f);
    struct NamedTensor *bv = Scalar(2.1f);
    printf("a:%f, b:%f, x:%f \n", *av->data, *bv->data, *xv->data);
    struct NamedTensor *exp_val = fx(xv, av, bv);
    struct NamedTensor *exp_grad = fdx(xv, av, bv);
    printf("EXPECTED val:%f, grad:%f\n", *exp_val->data, *exp_grad->data);

    ComputeNode *x = Variable(xv, "x");
    ComputeNode *fx = Pow(Add(Mul(x, Param(av, "a")), Param(bv, "b")), Constant(Scalar(2.0f)));
    printf("ACTUAL2 val: %f, grad:%f\n", *Forword(fx)->data, *Backword(x)->data);
}

int main(int argc, char *argv[]) {
    grad_test();
    minist();
    return 0;
}