#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include "../memory/mem.h"

struct DimensionDef *tensor_add_dimension(struct NamedTensor *tensor, struct DimensionDef *dimension) {
    tensor->dimension_nums++;
    if (tensor->dimensions == NULL) {
        tensor->dimensions = dimension;
    } else {
        dlist_append_tail(&tensor->dimensions->node, &dimension->node);
    }
    return dimension;
}

struct DimensionDef *Dimension(const char* name, uint64_t size) {
    DimensionDef *dimension = (DimensionDef*)AallocMem(sizeof(DimensionDef));
    if (dimension == NULL) {
        printf("dimension malloc failed!\n");
        exit(0);
    }
    dlist_init(&dimension->node);
    dimension->name = name;
    dimension->size = size;
    return dimension;
}

void tensor_print(struct NamedTensor *tensor) {
    if (tensor == NULL) {
        printf("NULL\n");
        return;
    }
    if (tensor->dimension_nums == 0) {
        printf("%f\n", *tensor->data);
    }
    if (tensor->dimensions == NULL) {
        printf("tensor's dimension should not be NULL!\n");
        exit(0);
    }
    if (tensor->data == NULL) {
        printf("tensor's data should not be NULL!\n");
        exit(0);
    }
    if (tensor->dimension_nums == 1) {
        printf("%s(%d) = [", tensor->dimensions->name, tensor->dimensions->size);
        for (size_t idx = 0; idx < tensor->dimensions->size; idx++) {
            if (idx == tensor->dimensions->size - 1) {
                printf("%f", tensor->data[idx]);
            } else {
                printf("%f, ", tensor->data[idx]);
            }
        }
        printf("]\n");
    }

    if (tensor->dimension_nums == 2) {
        DimensionDef *dimension1 = tensor->dimensions;
        DimensionDef *dimension2 = ContainerOf(tensor->dimensions->node.next, DimensionDef, node);

        printf("%s_%s(%dx%d) = [", dimension1->name, dimension2->name, dimension1->size, dimension2->size);
        for (size_t i = 0; i < dimension1->size; i++) {
            printf("[");
            for (size_t j = 0; j < dimension2->size; j++) {
                if (j == dimension2->size - 1) {
                    printf("%f", tensor->data[dimension2->size * i + j]);
                } else {
                    printf("%f, ", tensor->data[dimension2->size * i + j]);
                }
            }
            if (i == dimension1->size - 1) {
                printf("]");
            } else {
                printf("],");
            }
        }
        printf("]\n");
    }
}

struct NamedTensor *Tensor() {
    NamedTensor *tensor = (NamedTensor*)AallocMem(sizeof(NamedTensor));
    if (tensor == NULL) {
        printf("named tensor malloc failed!\n");
        exit(0);
    }

    tensor->data = NULL;
    tensor->dimension_nums = 0;
    tensor->dimensions = NULL;
    tensor->addDimension = tensor_add_dimension;
    tensor->print = tensor_print;

    return tensor;
}

struct NamedTensor *Scalar(double v) {
    NamedTensor *tensor = Tensor();
    double *val = AallocMem(sizeof(double));
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