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
#include "mm.h"


/* For bins*/
int Bin::init() {
    if ((free_list = (char *) heap.mem_sbrk(4 * WSIZE)) == (void *) -1)
        return -1;
    put((ADDRPTR) free_list, 0); /* Alignment padding */
    put((ADDRPTR) free_list + 1, pack(DSIZE, 1)); /*Prologue header*/
    put((ADDRPTR) free_list + 2, pack(DSIZE, 1)); /*Prologue footer*/
    put((ADDRPTR) free_list + 3, pack(0, 1)); /*Epilogue footer*/
    free_list += 2 * WSIZE;

    /*Extend the empty heap with a free block of chunksize */
    if (extend_heap(CHUNKSIZE / WSIZE) == nullptr)
        return -1;
    return 0;
}


mchunkptr Bin::extend_heap(SIZE_T words) {
    char *p;
    SIZE_T size;
    mchunkptr victim, nxtChunk;

    /*Allocate an even number of words to maintain alignment*/
    size = words % 2 ? (words + 1) * WSIZE : words * WSIZE;
    if ((long) (p = (char *) heap.mem_sbrk((int) size)) == -1)
        return nullptr;
    victim = reinterpret_cast<mchunkptr>(p);
    /*Initialize free block header/footer and the epilogue header*/
    victim->set(size, 0); // set free chunk header and free chunk footer
    nxtChunk = victim->nxtChunk();
    nxtChunk->set(0, 1); // new epilogue header

    /*Coalesce if the previous block  was free*/
    return coalesce(victim);
}

mchunkptr Bin::coalesce(mchunkptr p) {
    // get prev chunk's in_use bit and next chunk's in_use bit
    int prevInUse =


}
