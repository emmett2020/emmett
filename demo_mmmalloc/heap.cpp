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

#include "heap.h"

void *Heap::mem_sbrk(int incr) {
    char *old_brk = mem_brk;
    if ((incr < 0) || (mem_brk + incr > mem_max_addr)) {
        errno = ENOMEM;
        fprintf(stderr, "ERROR: mem_sbrk failed.Ran out of memory...\n");
        return (void *) -1;
    }
    mem_brk += incr;
    return (void *) old_brk;
}