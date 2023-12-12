#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>

struct Dimension *tensor_add_dimension(struct NamedTensor *tensor, struct Dimension *dimension) {
    tensor->dimension_nums++;
    // tensor->dimensions
}


struct DimensionDef *Dimension(const char* name, uint64_t size) {
    DimensionDef *dimension = (DimensionDef*)malloc(sizeof(DimensionDef));
    if (dimension == NULL) {
        printf("dimension malloc failed!\n");
        exit(0);
    }
    dimension->name = name;
    dimension->size = size;
    return dimension;
}

struct NamedTensor *Tensor() {
    NamedTensor *tensor = (NamedTensor*)malloc(sizeof(NamedTensor));
    if (tensor == NULL) {
        printf("named tensor malloc failed!\n");
        exit(0);
    }

    tensor->data = NULL;
    tensor->dimension_nums = 0;
    tensor->dimensions = NULL;
    tensor->addDimension = tensor_add_dimension;

    return tensor;
}

struct NamedTensor *Scalar(double v) {
    NamedTensor *tensor = Tensor();
    double *val = malloc(sizeof(double));
    if (val == NULL) {
        printf("scalar malloc failed!\n");
        exit(0);
    }
    *val = v;
    tensor->data = val;
    return tensor;
}

struct NamedTensor *Vector(struct DimensionDef *dimension, double *vector) {
    NamedTensor *tensor = Tensor();
    tensor->addDimension(tensor, dimension);
    tensor->data = vector;
    return tensor;
}

struct NamedTensor *Matrix(struct DimensionDef *dimension1, struct DimensionDef *dimension2, double *matrix) {
    NamedTensor *tensor = Tensor();
    tensor->addDimension(tensor, dimension1);
    tensor->addDimension(tensor, dimension2);
    tensor->data = matrix;
    return tensor;
}