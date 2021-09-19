/*
  --------------------------------------
     ╭╩══╮╔══════╗╔══════╗╔══════╗ 
    ╭    ╠╣      ╠╣      ╠╣      ╣
    ╰⊙══⊙╯╚◎════◎╝╚◎════◎╝╚◎════◎╝
  --------------------------------------
  @date:   2021-九月-18
  @author: xiaomingZhang2020@outlook.com
  --------------------------------------
*/

#include "gcmalloc.h"


void GC::scan_region(ADDRPTR beg, ADDRPTR end) {
    ADDRPTR p = beg;
    mchunkptr cp;
    while (p != end) {
        auto q = used_list();
        while (q != nullptr) {
            cp = q->chunkp;

            /* Could be traced */
            if ((char *) p >= cp->mem() && (char *) p < &(*(char *) (cp->footer()))) {
                q->marked = true;
                break;
            }
            q = q->next;
        }
        ++p;
    }

}


void GC::scan_heap() {
    char *hp;
    mAllocPtr p, q;
    mchunkptr pp, qp;

    /* O(n^3) */
    for (p = used_list(); p != nullptr; p = p->next) {
        if (!p->marked) // haven't been marked
            continue;
        pp = p->chunkp;
        for (hp = pp->mem(); hp < (char *) pp->footer(); ++hp) {
            for (q = used_list(); q != nullptr; q = q->next) {
                if (q == p)
                    continue;
                qp = q->chunkp;
                if (hp > qp->mem() && hp < (char *) qp->footer()) {
                    q->marked = true;
                }
            }
        }
    }


}

void GC::init() {
    FILE *statfp; // use to find stack bottom
    statfp = fopen("/proc/self/stat", "r");
    assert(statfp != nullptr);
    /* There may have bug when run on 64-bit os*/
    fscanf(statfp,
           "%*d %*s %*c %*d %*d %*d %*d %*d %*u "
           "%*lu %*lu %*lu %*lu %*lu %*lu %*ld %*ld "
           "%*ld %*ld %*ld %*ld %*llu %*lu %*ld "
           "%*lu %*lu %*lu %lu", &m_stack_bottom);
    fclose(statfp);

    m_usedListhead->next = nullptr;
}

void GC::collect() {
    mAllocPtr pre, p;
    if (!used_list()) // no need to collect
        return;

    /* Scan the BSS and initialized data segment */
    scan_dataSegement();

    /* Scan the stack */
    scan_stack();

    /* Scan the heap */
    scan_heap();

    /* And now we collect */
    for (pre = m_usedListhead, p = m_usedListhead->next; p != nullptr; pre = p, p = p->next) {
        /* The chunk hasn't been marked. Thus, it must be set free. */
        if (!p->marked) {
            pre->next = p->next;
            mbin.free(p->chunkp->mem());
            p = p->next;
        }
    }
}

void *GC::malloc(SIZE_T sz) {
    auto ret = mbin.malloc(sz);
    Allocator allocator;
    allocator.marked = false;
    allocator.chunkp = new Chunk(ret);
    allocator.next = m_usedListhead->next;
    m_usedListhead->next =  allocator.next;
    return ret;
}


void GC::scan_dataSegement() {
    extern char end, etext; // Provided by the linker
    scan_region((ADDRPTR) &etext, (ADDRPTR) &end);
}

void GC::scan_stack() {
    ADDRPTR stack_top;
    asm volatile ("movl %%ebp, %%0" : "=r"(stack_top));
    scan_region(stack_top, m_stack_bottom);
}
