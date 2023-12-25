#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include "../../tensorgrad/memory/mem.h"
#include "../../tensorgrad/common/random.h"
#include "../../tensorgrad/autograd/ops.h"
#include "../../tensorgrad/autograd/compute_node.h"
#include "../../tensorgrad/optimizer/adam.h"
#include "../../tensorgrad/lossfunc/cross_entropy.h"
#include "../../tensorgrad/autograd/tensor/tensor.h"

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
    struct NamedTensor *A = Matrix(Dimension("A", 10), Dimension("A", 10), Adata);
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
    ComputeNode *mulNode = Mul(paramA, ReLU(Variable(X, "X")));
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

    minNetTest();
    return 0;
}