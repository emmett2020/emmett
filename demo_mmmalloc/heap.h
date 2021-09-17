/*
  --------------------------------------
     ╭╩══╮╔══════╗╔══════╗╔══════╗
    ╭    ╠╣      ╠╣      ╠╣      ╣
    ╰⊙══⊙╯╚◎════◎╝╚◎════◎╝╚◎════◎╝
  --------------------------------------
  @date:   2021-九月-17
  @author: xiaomingZhang2020@outlook.com
  --------------------------------------
*/

#ifndef DEMO_HEAP_H
#define DEMO_HEAP_H

#include <unistd.h>
#include <sys/mman.h>
#include <cerrno>
#include <iostream>

/*
    SIZE_T - Header's or footer's size.
             Since we always want 32 bit, we defined SIZE_T as unsigned int
 */
#ifndef SIZE_T
#define SIZE_T unsigned
#endif

class Heap {
public:
    /* Heap - Initialize the memory system model */
    Heap() {
        /* Use glibc malloc to simulate mmap */
        m_mem_heap = static_cast<char *> (malloc(HEAP_MAX_SIZE));
        m_mem_brk = static_cast<char *> (m_mem_heap);
        m_mem_max_addr = static_cast<char *> (m_mem_heap + HEAP_MAX_SIZE);
    }

/*
    mem-sbrk - Simple model of the sbrk function. Extends the heap br incr bytes
               and return the start address of the new area. In this model, the heap
               cannot be shrink.
 */
    void *mem_sbrk(int incr);

private:
    /* private global variables */
    char *m_mem_heap;      // points to first byte of heap
    char *m_mem_brk;       // points to last byte of heap plus 1
    char *m_mem_max_addr;  // max legal heap addr plus 1
private:
    const int HEAP_MAX_SIZE = 1024 * 1024;

};

// there will be changed in the future
extern Heap heap;

#endif //DEMO_HEAP_H
