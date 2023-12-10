#ifndef __DLIST_H__
#define __DLIST_H__

#include <stddef.h>

typedef struct ListNode {
	struct ListNode *prev;
	struct ListNode *next;
} ListNode;

#define OffsetOf(type, member) ((int)((char *)&((type *)0)->member))
#define ContainerOf(ptr, type, member) \
  ((type *)((char *)(ptr) - (char *)(&(((type *)0)->member))))

static inline void dlist_init(struct ListNode *node) {
	node->next = NULL;
	node->prev = NULL;
}

static inline void dlist_append_tail(struct ListNode *dist, struct ListNode *item) {
	ListNode *node = dist;
	if (node == NULL) {
		return;
	}
	while (node->next != NULL) {
		node = node->next;
	}
	node->next = item;
	item->prev = node;
}

static inline void dlist_insert(struct ListNode *dist, struct ListNode *item) {
	if (dist == NULL) {
		return;
	}
	if (dist->next == NULL) {
		dist->next = item;
		item->prev = dist;
		return;
	}
	if (dist->next != NULL) {
		item->next = dist->next;
		dist->next->prev = item;
		dist->next = item;
		item->prev = dist;
	}
}

#endif /* __DLIST_H__ */