#ifndef __RANDOM_H__
#define __RANDOM_H__

#include <stdlib.h>

static inline float frand(float max) {
    return ((float)rand() / RAND_MAX) * (max);
}

#endif /* __RANDOM_H__ */