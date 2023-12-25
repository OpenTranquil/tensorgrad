#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include "../../memory/mem.h"

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
    DimensionDef *dimension = (DimensionDef*)AllocMem(sizeof(DimensionDef));
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
    if (tensor->type == TENSOR_TYPE_SCALAR) {
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
    if (tensor->type == TENSOR_TYPE_ROW_VECTOR) {
        DimensionDef *width = ContainerOf(tensor->dimensions->node.next, DimensionDef, node);
        printf("%s(%d) = [", width->name, width->size);
        for (size_t idx = 0; idx < width->size; idx++) {
            if (idx == width->size - 1) {
                printf("%f", tensor->data[idx]);
            } else {
                printf("%f, ", tensor->data[idx]);
            }
        }
        printf("]\n");
    }

    if (tensor->type == TENSOR_TYPE_COLUMN_VECTOR) {
        DimensionDef *height = tensor->dimensions;
        DimensionDef *width = ContainerOf(tensor->dimensions->node.next, DimensionDef, node);
        printf("%s(%d) = [\n", height->name, height->size);
        for (size_t idx = 0; idx < height->size; idx++) {
            if (idx == height->size - 1) {
                printf("    %f\n", tensor->data[idx]);
            } else {
                printf("    %f,\n", tensor->data[idx]);
            }
        }
        printf("]\n");
    }

    if (tensor->type == TENSOR_TYPE_MATRIX) {
        DimensionDef *dimension1 = tensor->dimensions;
        DimensionDef *dimension2 = ContainerOf(tensor->dimensions->node.next, DimensionDef, node);

        printf("%s_%s(%dx%d) = [\n", dimension1->name, dimension2->name, dimension1->size, dimension2->size);
        for (size_t i = 0; i < dimension1->size; i++) {
            printf("    [");
            for (size_t j = 0; j < dimension2->size; j++) {
                if (j == dimension2->size - 1) {
                    printf("%f", tensor->data[dimension2->size * i + j]);
                } else {
                    printf("%f, ", tensor->data[dimension2->size * i + j]);
                }
            }
            if (i == dimension1->size - 1) {
                printf("]\n");
            } else {
                printf("],\n");
            }
        }
        printf("]\n");
    }
}

struct NamedTensor *Tensor() {
    NamedTensor *tensor = (NamedTensor*)AllocMem(sizeof(NamedTensor));
    if (tensor == NULL) {
        printf("named tensor malloc failed!\n");
        exit(0);
    }

    tensor->data = NULL;
    tensor->type = TENSOR_TYPE_TENSOR;
    tensor->dimension_nums = 0;
    tensor->dimensions = NULL;
    tensor->addDimension = tensor_add_dimension;
    tensor->print = tensor_print;

    return tensor;
}

struct NamedTensor *Scalar(double v) {
    NamedTensor *tensor = Tensor();
    double *val = AllocMem(sizeof(double));
    if (val == NULL) {
        printf("scalar malloc failed!\n");
        exit(0);
    }
    *val = v;
    tensor->data = val;
    tensor->type = TENSOR_TYPE_SCALAR;
    return tensor;
}

struct NamedTensor *ColumnVector(struct DimensionDef *dimension, double *vector) {
    NamedTensor *tensor = Tensor();
    // n * 1
    tensor->addDimension(tensor, dimension);
    tensor->addDimension(tensor, Dimension(dimension->name, 1));
    tensor->type = TENSOR_TYPE_COLUMN_VECTOR;
    tensor->data = vector;
    return tensor;
}

struct NamedTensor *RowVector(struct DimensionDef *dimension, double *vector) {
    NamedTensor *tensor = Tensor();
    // 1 * n
    tensor->addDimension(tensor, Dimension(dimension->name, 1));
    tensor->addDimension(tensor, dimension);
    tensor->type = TENSOR_TYPE_ROW_VECTOR;
    tensor->data = vector;
    return tensor;
}

struct NamedTensor *Vector(struct DimensionDef *dimension, double *vector) {
    return ColumnVector(dimension, vector);
}

struct NamedTensor *Matrix(struct DimensionDef *dimension1, struct DimensionDef *dimension2, double *matrix) {
    NamedTensor *tensor = Tensor();
    tensor->addDimension(tensor, dimension1);
    tensor->addDimension(tensor, dimension2);
    tensor->type = TENSOR_TYPE_MATRIX;
    tensor->data = matrix;
    return tensor;
}