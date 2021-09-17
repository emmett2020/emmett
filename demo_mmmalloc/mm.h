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

#ifndef DEMO_MM_H
#define DEMO_MM_H
#include "heap.h"

/* ADDRPTR - Address pointer to */
#define ADDRPTR unsigned int*

const SIZE_T WSIZE = 4;            // Word and header/footer size (bytes)
const SIZE_T DSIZE = 8;            // Double word size (bytes)
const SIZE_T CHUNKSIZE = 1 << 12;  // Extend heap by this amount, equals to 4KB


/* pack - Pack a size and allocated a bit into a word */
inline SIZE_T pack(SIZE_T size, int flag) { return size | flag; }

/* get & put - Read and write a word at address p */
inline SIZE_T get(const ADDRPTR p) { return *p; }
inline void put(ADDRPTR p, SIZE_T val) { *p = val; }


/*
 * Manage the real memory chunk.The chunk looks like this:

   lower address
   header: [    1B    ] [    1B    ] [    1B    ] [---- ---a]  <- size + allocated flag
      mem: [          ] [          ] [          ] [         ]  <- real allocated
                                    ...
                                    ...
           [          ] [          ] [          ] [         ]
   footer: [    1B    ] [    1B    ] [    1B    ] [---- ---a]  <- size + allocated flag
   nxtchunk's header: [          ] [          ] [..]   [..]
   higher address

 * What's the size stands for?
    there is one difference in size compared to glibc2.
    the size in header stands for the sum of it's size
    and it's footer and nextChunk's headers.
*/
class Chunk;
using mchunkptr = Chunk *;


class Chunk {
public:
    Chunk(void *p) {
        m_mem = (char *) p;
    }

    Chunk() : m_mem(nullptr) {}

public:
    /*Given chunk ptr cp, compute address of its header and footer*/
    ADDRPTR header() { return reinterpret_cast<ADDRPTR> (m_mem - WSIZE); }
    ADDRPTR footer() { return reinterpret_cast<ADDRPTR> (m_mem + size() - DSIZE); }

    /*Read the size and allocated fields from address p*/
    SIZE_T size() { return  *header() & ~0x7; }
    SIZE_T inUseFlag() { return *header() & ~0x1; }

    /*set - Set new size and flag for header and footer*/
    void set(SIZE_T sz, int flag) {
        // when we change size, we must change our footer
        *header() = pack(sz, flag);
        auto footer = reinterpret_cast<ADDRPTR> (m_mem + sz - DSIZE);
        *footer = pack(sz, flag);
    }

    /*Compute address of next and previous chunks*/
    mchunkptr nxtChunk() {
        return reinterpret_cast<mchunkptr>(m_mem + size());
    }

    mchunkptr preChunk() {
        size_t prev_size = get((ADDRPTR) (m_mem) - DSIZE) & ~0x7;
        return reinterpret_cast<mchunkptr> (m_mem - prev_size);
    }

private:
    char *m_mem;
};


/*
 * To be simplified, there is only one free_list.
 * The free_list looks like this:

   lower address( [] == 1 bytes )
   [    0    ] [    0    ] [    0    ] [    0    ]  <-- pad
   [    0    ] [    0    ] [    0    ] [0000 1001]  <-- header == 0x8 | 0x1
   [    0    ] [    0    ] [    0    ] [0000 1001]  <-- footer == 0x8 | 0x1
   [    0    ] [    0    ] [    0    ] [0000 0001]  <-- Epilogue footer
   higher address

 * Why do we put the free_list at free_list + 2 * WSIZE?
     In the initial situation, free_list is in the end of header,
     we can treat it as a chunk.
     Then we can find the chunk size in header(free_list - WSIZE), and it is 8 bytes.
     Note that chunk's size include this chunk's footer and nextChunk's header.
     If there have new chunk allocated, we could find nextChunk by free_list + 8.

 * What's epilogue footer?
    We use this to mark the end of free_list. If chunk is at the end of free_list,
    the nextChunk's header will be epilogue footer though next chunk isn't exists.
    And when we extend the free_list, the epilogue footer will be rewrite by new
    chunk, and it will stands for new chunk's header.

 */

class Bin {
public:
    /* Create the initial empty heap */
    int init();

    /*Equals to ptmalloc-free*/
    void free(void *p);

    /*
     * malloc - Equals to ptmalloc-malloc。
     *      The minimal size is 4 * WSIZE. Which includes header, footer, and pad.
     *      The size of pad is 2 * WSIZE since all address are aligned to two WSIZE.
    */
    void *malloc(SIZE_T size);

private:
    /*
     * extend_heap - This will be used in two case.
     *               Case1: initialize heap
     *               Case2: mm_malloc can't find a suitable chunk
    */
    mchunkptr extend_heap(SIZE_T words);
    mchunkptr coalesce(mchunkptr p);
    mchunkptr find_fit(SIZE_T asize);

    /*
     * place - Place block of asize bytes at start of free block bp
     *         and split if remainder would be at least minimum block size
    */
    void place(mchunkptr p, SIZE_T asize);

private:
    char *free_list;
};


#endif //DEMO_MM_H
