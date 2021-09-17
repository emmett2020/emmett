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

#include "mm.h"


/* For bins*/
int Bin::init() {
    ADDRPTR p; //
    free_list = static_cast<char *>(heap.mem_sbrk(4 * WSIZE));
    if (free_list == reinterpret_cast<char *> (-1))
        return -1;
    p = reinterpret_cast<ADDRPTR> (free_list);
    put(p, 0); // Alignment padding
    put(p + 1, pack(DSIZE, 1)); //  Prologue header
    put(p + 2, pack(DSIZE, 1)); //  Prologue footer
    put(p + 3, pack(0, 1)); // Epilogue footer
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
    size_t prevInUse = p->preChunk()->inUseFlag();
    size_t nxtInUse = p->nxtChunk()->inUseFlag();

    if (prevInUse && nxtInUse) // no need to coalesce
        return p;
    else if (prevInUse && !nxtInUse) { // coalesce the next chunk
        size_t nextSize = p->nxtChunk()->size();
        p->set(p->size() + nextSize, 0);
        return p;
    } else if (!prevInUse && nxtInUse) { // coalesce the prev chunk
        size_t prevSize = p->preChunk()->size();
        p->preChunk()->set(p->size() + prevSize, 0);
        return p->preChunk();
    } else { // coalesce the prev and nxt chunke
        size_t prevSize = p->preChunk()->size();
        size_t nxtSize = p->nxtChunk()->size();
        p->preChunk()->set(p->size() + prevSize + nxtSize, 0);
        return p->preChunk();
    }
}

void Bin::free(void *p) {
    auto victim = static_cast<mchunkptr> (p);
    victim->set(victim->size(), 0); // don't modify the size field
    coalesce(victim);
}

void *Bin::malloc(SIZE_T size) {
    SIZE_T asize;       // Adjusted chunk size
    SIZE_T extendsize;  // Amount to extend heap if no fit
    mchunkptr victim;

    /*Ignore spurious request*/
    if (size == 0)
        return nullptr;

    /*Adjust block size to include overhead and alignment reqs*/
    if (size <= DSIZE) // minimal size
        asize = 2 * DSIZE;
    else // align to 2 * DSIZE
        asize = (size + DSIZE - 1) & ~(DSIZE - 1);

    /*Search the free_list for a fit*/
    if ((victim = find_fit(asize)) != nullptr) {
        place(victim, asize);
        return static_cast<void *> (victim);
    }

    /*No fit found. Get more memory and place the block*/
    extendsize = asize > CHUNKSIZE ? asize : CHUNKSIZE;
    if ((victim = extend_heap(extendsize)) == nullptr) {
        return nullptr;
    }

    place(victim, asize);
    return static_cast<void *>(victim);
}

mchunkptr Bin::find_fit(SIZE_T asize) {
    auto p = reinterpret_cast<mchunkptr> (free_list);
    while (p->size() > 0) {
        // WARNING: this is different from book. But I consider that I was right.
        if (p->size() - DSIZE >= asize && p->inUseFlag() == 0) {
            return p;
        }
        p = p->nxtChunk();
    }
    return nullptr;
}

/*
 * f == footer, h == header, r == remainder
 *           h       p                                       f             nxtChunk
 *           | WSIZE |                    p->size()                           |
 *           | WSIZE |      allocated_size         |      remainder_size      |
 *           [ WSIZE |    asize    | WSIZE | WSIZE |         | WSIZE  | WSIZE ]
 *           h       p             f       h       r         f        h    nxtChunk
 */

void Bin::place(mchunkptr p, SIZE_T asize) {
    size_t sz = p->size();
    if (sz - asize >= 2 * DSIZE) {
        SIZE_T allocated_size = asize + DSIZE;
        SIZE_T remainder_size = sz - allocated_size;

        p->set(allocated_size, 1); // allocate
        auto remainder = (mchunkptr) ((char *) p + allocated_size);
        remainder->set(remainder_size, 0);
    } else {
        p->set(sz, 1);
    }
}



