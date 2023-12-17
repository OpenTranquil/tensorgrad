#ifndef __MEM_H__
#define __MEM_H__

#include <stddef.h>

void *AallocMem(size_t size);
void memfree(void *mem);

#endif /* __MEM_H__ */