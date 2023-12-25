#ifndef __AUTOGRAD_OPS_H__
#define __AUTOGRAD_OPS_H__

#include "compute_node.h"

struct NamedTensor *forword(ComputeNode *node);
struct NamedTensor *backward(ComputeNode *node);

#endif /* __AUTOGRAD_OPS_H__ */