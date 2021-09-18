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

#include "heap.h"

Heap heap = Heap();
void *Heap::mem_sbrk(int incr) {
    char *old_brk = m_mem_brk;
    if ((incr < 0) || (m_mem_brk + incr > m_mem_max_addr)) {
        errno = ENOMEM;
        fprintf(stderr, "ERROR: mem_sbrk failed. Ran out of memory...\n");
        return nullptr;
    }
    m_mem_brk += incr;
    return static_cast<void *> (old_brk);
}