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

#ifndef DEMO_MM_H
#define DEMO_MM_H

/*
 The address will be align to 2 * WSIZE = 8 bytes.
 */
#ifndef ADDRPTR
#define ADDRPTR unsigned int *
#endif


const int WSIZE = 4;            // Word and header/footer size (bytes)
const int DSIZE = 8;            // Double word size (bytes)
const int CHUNKSIZE = 1 << 12;  // Extend heap by this amount (bytes), equals 4KB


/* Pack a size and allocated a bit into a word*/
inline SIZE_T pack(SIZE_T size, int flag) { return size | flag; }


/* Read and write a word at address p*/
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
   next chunk's header:
   higher address

 * What's size stands for?
    there is one difference in size compared to glibc2.
    the size in header stands for the sum of it's size
    and it's footer and nextChunk's headers.

*/
class Chunk;

using mchunkptr = Chunk *;

/*******************************************************************/

class Chunk {
public:
    Chunk(void *p) {
        m_mem = (char *) p;
    }

    Chunk() : m_mem(nullptr) {}

public:
    /*Given chunk ptr cp, compute address of its header and footer*/
    ADDRPTR header() { return (ADDRPTR) (m_mem - WSIZE); }
    ADDRPTR footer() { return (ADDRPTR) (m_mem + size() - DSIZE); }

    /*Read the size and allocated fields from address p*/
    SIZE_T size() { return *header() & ~0x7; }
    SIZE_T inUseFlag() { return *header() & ~0x1; }

    /*set new size and flag*/
    void set(SIZE_T sz, int flag) {
        // when we change size, we must change our footer
        *header() = pack(sz, flag);
        ADDRPTR footer = (ADDRPTR) (m_mem + sz - DSIZE);
        *footer = pack(sz, flag);
    }

    /*Given chunk ptr bp, compute address of next and previous blocks*/
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

private:
    /*
     * This will be used in two case.
     * Case1: initialize heap
     * Case2: mm_malloc can't find a suitable chunk
    */
    mchunkptr extend_heap(SIZE_T words);
    mchunkptr coalesce(mchunkptr p);


private:
    static char *free_list;
};


#endif //DEMO_MM_H
