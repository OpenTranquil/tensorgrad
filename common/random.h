#ifndef __RANDOM_H__
#define __RANDOM_H__

#include <stdlib.h>
#include <time.h>

static inline float frand(float max) {
    srand((unsigned int)time(NULL));
    return ((float)rand() / RAND_MAX) * (max);
}

#endif /* __RANDOM_H__ */