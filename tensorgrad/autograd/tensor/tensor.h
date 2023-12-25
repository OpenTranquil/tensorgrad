#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <stdint.h>
#include "../../common/dlist.h"



typedef enum TensorType {
    TENSOR_TYPE_SCALAR,
    TENSOR_TYPE_ROW_VECTOR,
    TENSOR_TYPE_COLUMN_VECTOR,
    TENSOR_TYPE_MATRIX,
    TENSOR_TYPE_TENSOR,
    TENSOR_TYPE_MAX,
} TensorType;

static char *TENSOR_TYPE_STRS[TENSOR_TYPE_MAX] = {
    "Scalar", "Row Vector", "Column Vector", "Matrix", "Tensor", "NONE"
};

static const char *TensorTypeName(TensorType type) {
    return TENSOR_TYPE_STRS[type];
}

typedef struct DimensionDef {
    struct ListNode node;
    const char *name;
    uint64_t size;
} DimensionDef;

typedef struct DimensionDef *(*TensorAddDimension)(struct NamedTensor *tensor, struct DimensionDef *dimension);
typedef void (*TensorPrint)(struct NamedTensor *tensor);

typedef struct NamedTensor {
    TensorType type;
    TensorAddDimension addDimension;
    TensorPrint print;
    uint64_t dimension_nums;
    struct DimensionDef *dimensions;
    double *data;
} NamedTensor;

struct DimensionDef *Dimension(const char* name, uint64_t size);
struct NamedTensor *Tensor();
struct NamedTensor *Scalar(double v);
struct NamedTensor *Vector(struct DimensionDef *dimension, double *vector);
struct NamedTensor *ColumnVector(struct DimensionDef *dimension, double *vector);
struct NamedTensor *RowVector(struct DimensionDef *dimension, double *vector);
struct NamedTensor *Matrix(struct DimensionDef *dimension1, struct DimensionDef *dimension2, double *matrix);

#endif /* __TENSOR_H__ */