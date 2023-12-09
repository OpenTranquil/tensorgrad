#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "autograd/grad.h"
#include "model/layer.h"
#include "model/layers/conv2d.h"
#include "model/layers/maxpooling2d.h"
#include "model/layers/flatten.h"
#include "model/layers/dense.h"
#include "model/model.h"
#include "tensor/tensor.h"

// f = (ax + b)^2
// Z = ax
// Y = Z + b
// F = Y ^ 2

// fx' = FY'YZ'Zx'
// fx' = 2Y * 1 * a
// fx' = 2(ax +b) * 1 * a
// fx' = 2(ax+b)a

double fx(double x, double a, double b) {
    double val = pow((a * x + b), 2);
    return val;
}

double fdx(double x, double a, double b) {
    double grad = 2.0f * (a * x + b) * a;
    return grad;
}

void minist() {
    NamedTensor *tensor = Tensor();
    tensor->addDimension(tensor, Dimension("batch_size", 28));
    tensor->addDimension(tensor, Dimension("height", 28));
    tensor->addDimension(tensor, Dimension("weight", 28));

    NNModel *model = SequentialModel();
    model->addLayer(model, Conv2D(32, Tuple(3, 3), RELU));
    model->addLayer(model, MaxPooling2D(Tuple(2,2)));
    model->addLayer(model, Conv2D(64, Tuple(3, 3), RELU));
    model->addLayer(model, MaxPooling2D(Tuple(2,2)));
    model->addLayer(model, Conv2D(64, Tuple(3, 3), RELU));
    model->addLayer(model, Flatten());
    model->addLayer(model, Dense(64, RELU));
    model->addLayer(model, Dense(10, SOFTMAX));

    uint64_t epochs = 5;
    uint64_t batch_size = 64;
    float validation_split = 0.2;

    model->compile(model, ADAM, CROSS_ENTROPY);
    model->fit(model, tensor, epochs, batch_size, validation_split);
    model->evaluate(model);
}

int main(int argc, char *argv[]) {
    double av = 5.1f;
    double xv = 6.3f;
    double bv = 2.1f;
    printf("a:%f, b:%f, x:%f \n", av, bv, xv);
    double exp_val = fx(xv, av, bv);
    double exp_grad = fdx(xv, av, bv);
    printf("EXPECTED val:%f, grad:%f\n", exp_val, exp_grad);

    ComputeNode *x = Variable(xv, "x");
    ComputeNode *fx = Pow(Add(Mul(x, Param(av, "a")), Param(bv, "b")), Constant(2.0f));
    printf("ACTUAL2 val: %f, grad:%f\n", Forword(fx), Backword(x));


    minist();
    return 0;
}