#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include "memory/mem.h"
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
    float *data = AllocMem(sizeof(float) * 28 * 28);
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
    // model->addLayer(model, Conv2D(32, Tuple(3, 3), ACTV_RELU));
    // model->addLayer(model, MaxPooling2D(Tuple(2,2), ACTV_NONE));
    // model->addLayer(model, Conv2D(64, Tuple(3, 3), ACTV_RELU));
    // model->addLayer(model, MaxPooling2D(Tuple(2,2), ACTV_NONE));
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

//       pow
//       / \
//      +   2
//     / \
//    *   b
//   / \
//  a   x
void grad_test() {
    struct NamedTensor *av = Scalar(5.1f);
    struct NamedTensor *xv = Scalar(6.3f);
    struct NamedTensor *bv = Scalar(2.1f);
    printf("a:%f, b:%f, x:%f \n", *av->data, *bv->data, *xv->data);
    struct NamedTensor *exp_val = fx(xv, av, bv);
    struct NamedTensor *exp_grad = fdx(xv, av, bv);
    printf("EXPECTED val:%f, grad:%f\n", *exp_val->data, *exp_grad->data); 

    ComputeNode *x = Param(xv, "x");
    ComputeNode *fx = Pow(Add(Mul(x, Variable(av, "a")), Variable(bv, "b")), Constant(Scalar(2.0f)));
    printf("ACTUAL2 val: %f, grad:%f\n", *Forword(fx)->data, *Backword(x)->data);
}

//F = SoftMax(A * ReLU(X)  + B);
//F1 = A * ReLU(X) + B
//F2 = A * ReLU(X)
//F3 = ReLU(X)
//F = SoftMax(F1)
//F1 = F2 + B
//F2 = A * F3
//F'A = F'F1 * F1'F2 * F2'A
void minNetTest() {
    double *Xdata = (double *)AllocMem(sizeof(double) * 10);
    for (size_t i = 0; i < 10; i++) {
        Xdata[i] = frand(10.0f);
    }
    struct NamedTensor *X = Vector(Dimension("X", 10), Xdata);
    X->print(X);

    double *Adata = (double *)AllocMem(sizeof(double) * 10 * 10);
    for (size_t i = 0; i < 10; i++) {
        for (size_t j = 0; j < 10; j++) {
            Adata[i * 10 + j] = frand(10.0f);
        }
    }
    struct NamedTensor *A = Matrix(Dimension("H", 10), Dimension("W", 10), Adata);
    A->print(A);

    double *Bdata = (double *)AllocMem(sizeof(double) * 10 * 10);
    for (size_t i = 0; i < 10; i++) {
        Bdata[i] = frand(10.0f);
    }
    struct NamedTensor *B = ColumnVector(Dimension("B", 10), Bdata);
    B->print(B);


    ComputeNode *reluNode = ReLU(Variable(X, "X"));
    NamedTensor *reluTensor = Forword(reluNode);
    reluTensor->print(reluTensor);

    ComputeNode *paramA = Param(A, "A");
    ComputeNode *mulNode = Mul(ReLU(Variable(X, "X")), paramA);
    NamedTensor *mulTensor = Forword(mulNode);
    mulTensor->print(mulTensor);

    ComputeNode *paramB = Param(B, "B");
    ComputeNode *addNode = Add(Mul(paramA, ReLU(Variable(X, "X"))), paramB);
    NamedTensor *addTensor = Forword(addNode);
    addTensor->print(addTensor);

    ComputeNode *softmaxNode = Softmax(Add(Mul(paramA, ReLU(Variable(X, "X"))), paramB));
    struct NamedTensor *probVector = Forword(softmaxNode);
    probVector->print(probVector);

    double *expectedData = (double *)AllocMem(sizeof(double) * 10);
    for (size_t i = 0; i < 10; i++) {
        expectedData[i] = 0.0f;
    }
    expectedData[4] = 1.0f;
    NamedTensor *expectedVec = RowVector(Dimension("expected", 10), expectedData);
    expectedVec->print(expectedVec);

    LossFunc *lossFunc = CrossEntropyLossFunc();
    double loss = lossFunc->ops.forword(lossFunc, probVector, expectedVec);
    printf("LOSS: %f\n", loss);

    Backword(paramA);
}

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));

    grad_test();
    minNetTest();
    // minist();
    return 0;
}