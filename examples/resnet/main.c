#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include "../../tensorgrad/memory/mem.h"
#include "../../tensorgrad/common/random.h"
#include "../../tensorgrad/autograd/tensor/tensor.h"
#include "../../tensorgrad/autograd/ops.h"
#include "../../tensorgrad/autograd/compute_node.h"
#include "../../tensorgrad/optimizer/adam.h"
#include "../../tensorgrad/lossfunc/cross_entropy.h"
#include "model/layer.h"
#include "model/layers/conv2d.h"
#include "model/layers/maxpooling2d.h"
#include "model/layers/flatten.h"
#include "model/layers/dense.h"
#include "model/model.h"

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

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));

    minist();
    return 0;
}