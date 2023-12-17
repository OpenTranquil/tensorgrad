#include "mem.h"
#include <stddef.h>
#include <stdlib.h>

uint64_t __MemUsed = 0;

void *AllocMem(size_t size) {
    void *mem = malloc(size);
    if (mem != NULL) {
        __MemUsed += size;
    }
    return mem;
}

void memfree(void *mem) {
    free(mem);
}