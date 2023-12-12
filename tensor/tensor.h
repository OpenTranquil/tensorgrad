#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <stdint.h>

typedef struct DimensionDef {
    const char *name;
    uint64_t size;
} DimensionDef;

typedef struct DimensionDef *(*TensorAddDimension)(struct NamedTensor *tensor, struct DimensionDef *dimension);

typedef struct NamedTensor {
    TensorAddDimension addDimension;
    uint64_t dimension_nums;
    struct Dimension *dimensions;
    double *data;
} NamedTensor;

struct DimensionDef *Dimension(const char* name, uint64_t size);
struct NamedTensor *Tensor();
struct NamedTensor *Scalar(double v);
struct NamedTensor *Vector(struct DimensionDef *dimension, double *vector);
struct NamedTensor *Matrix(struct DimensionDef *dimension1, struct DimensionDef *dimension2, double *matrix);


#endif /* __TENSOR_H__ */