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

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));

    grad_test();
    return 0;
}