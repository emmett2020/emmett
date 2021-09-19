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

// WARNING: this version haven't been tested OK

#ifndef DEMO_GCMALLOC_H
#define DEMO_GCMALLOC_H
#include "../demo_mmmalloc/mm.h"

struct Allocator {
    bool marked;                // whether access
    mchunkptr chunkp;           // allocated chunk address
    Allocator *next;          // single list
    Allocator() : marked(false), chunkp(nullptr), next(nullptr) {}
};
using mAllocPtr = Allocator *;


class GC {
public:
    /* Mark & sweep */
    void collect();

    /* Wrapper of malloc */
    void *malloc(SIZE_T sz);

    GC() { init(); }

private:

    /* Find the absolute bottom of the stack and set stuff up.*/
    void init();

    /*
     * Scan a region of memory and mark any items in the used list appropriately.
     * Both arguments should be word aligned.
     * We will scan the BSS, the used chunks and the stack.
     */

    void scan_region(ADDRPTR beg, ADDRPTR end);


    /*
     * We must supposed to the memory is contiguous now. This should be improved
     * in the future.
     *
     * Scan the marked blocks for references to other unmarked blocks.
     */

    void scan_heap();


    /*
     *  etext - the address of etext is the last address past the text segment.
                The initialized data segment immediately follows the text segment
                and thus the address of etext is the start of the initialized data segment.

     *   end -  the address of end is the start of the heap, or the last address past
                the end of the BSS.

        Since there is no segment between the BSS and initialized segments,
        we don't have to treat them as separate entities and
        can scan them by iterating from &etext to &end.
     */

    void scan_dataSegement();


    /*
        %esp register - The top of the stack.
        %rbp register - the base of cur stack frame

        To be honest, I'm not an expert on finding the bottom of the stack,
        but I have a few rather poor ideas on how you can make an accurate.
        One possible way is you could scan the call stack for the env pointer,
        which would be passed as an argument to main.
        Another way would be to start at the top of the stack and read every
        subsequent address greater and handling the inexorable SIGSEGV.
        But we're not going to do it either way.
        Instead, we're going to exploit the fact that Linux puts the bottom of
        the stack in a string in a file in the process's entry in the proc
        directory (phew!). This sounds silly and terribly indirect. Fortunately,
        I don't feel ridiculous for doing doing it because it's literally the exact
        same thing Boehm GC does to find the bottom of the stack!

     */

    void scan_stack();

    /* mark and sweep */
    void mark();
    void sweep();
    static mAllocPtr used_list() { return m_usedListhead->next; }

private:
    static mAllocPtr m_usedListhead;
    ADDRPTR m_stack_bottom;
    Bin mbin; // used for malloc and free
};


#endif //DEMO_GCMALLOC_H
