#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <stdint.h>

typedef struct DimensionDef {
    const char *name;
    uint64_t size;
} DimensionDef;

typedef struct DimensionDef *(*TensorAddDimension)(struct NamedTensor *tensor, struct DimensionDef *dimension);

typedef struct NamedTensor {
    uint64_t dimension_nums;
    struct Dimension *dimensions;
    double *data;

    TensorAddDimension addDimension;
} NamedTensor;

struct DimensionDef *Dimension(const char* name, uint64_t size);
struct NamedTensor *Tensor();

#endif /* __TENSOR_H__ */