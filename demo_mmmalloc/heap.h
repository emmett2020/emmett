/*
  --------------------------------------
     ╭╩══╮╔══════╗╔══════╗╔══════╗
    ╭    ╠╣      ╠╣      ╠╣      ╣
    ╰⊙══⊙╯╚◎════◎╝╚◎════◎╝╚◎════◎╝
  --------------------------------------
  @date: 2021-九月-17
  @author: xiaomingZhang2020@outlook.com
  --------------------------------------
*/

#ifndef DEMO_HEAP_H
#define DEMO_HEAP_H

#include <unistd.h>
#include <sys/mman.h>

/*
 * Header's or footer's size. Since in some system, size_t == 4bytes,
 * while 8 bytes in others'.
 */
#ifndef SIZE_T
#define SIZE_T size_t
#endif



// Manage the heap use single mode
class Heap {
public:
    /* mem_init - Initialize the memory system modelx */
    Heap() {
        mem_heap = (char *) malloc(HEAP_MAX_SIZE); // use glibc malloc
        mem_brk = (char *) mem_heap;
        mem_max_addr = (char *) (mem_heap + HEAP_MAX_SIZE);
    }

    /*
     * Simple model of the sbrk function. Extends the heap br incr bytes
     * and return the start address of the new area. In this model, the heap
     * cannot be shrink.
     */
    void *mem_sbrk(int incr);


private:
    /* private global variables */
    static char *mem_heap;      // points to first byte of heap
    static char *mem_brk;       // points to last byte of heap plus 1
    static char *mem_max_addr;  // max legal heap addr plus 1
private:
    const int HEAP_MAX_SIZE = 1024 * 1024;

};

// there will be changed in the future
static Heap heap;

#endif //DEMO_HEAP_H
