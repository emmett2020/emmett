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
    ADDRPTR p;
    free_list = static_cast<char *>(heap.mem_sbrk(4 * WSIZE));
    if (free_list == reinterpret_cast<char *> (-1))
        return -1;
    p = reinterpret_cast<ADDRPTR> (free_list);
    put(p, 0); // Alignment padding
    put(p + 1, pack(DSIZE, 1)); //  Prologue header
    put(p + 2, pack(DSIZE, 1)); //  Prologue footer
    put(p + 3, pack(0, 0x1)); // Epilogue footer
    free_list += 2 * WSIZE;

    /*Extend the empty heap with a free chunk of chunk size */
    if (extend_heap(CHUNKSIZE / WSIZE) == nullptr)
        return -1;
    return 0;
}


char *Bin::extend_heap(SIZE_T words) {
    char *p;
    SIZE_T size;
    /*Allocate an even number of words to maintain alignment*/
    size = words % 2 ? (words + 1) * WSIZE : words * WSIZE;
    p = static_cast<char *>(heap.mem_sbrk(size));
    if (!p)
        return nullptr;

    Chunk victim(p);
    /*Initialize free block header/footer and the epilogue header*/
    victim.set(size, 0); // set free chunk header and free chunk footer

    // Note that epilogue don't have footer, we mute not use set
    put((ADDRPTR) (p + size - WSIZE), pack(0, 1)); // new epilogue header

    /*Coalesce if the previous chunk  was free*/
    return coalesce(victim.mem());
}

char *Bin::coalesce(char *p) {
    Chunk chunk(p);
    // get prev chunk's in_use bit and next chunk's in_use bit
    auto preChunk = chunk.preChunk();
    auto nxtChunk = chunk.nxtChunk();
    size_t preSize = preChunk.size();
    size_t nxtSize = nxtChunk.size();
    size_t preInUse = preChunk.inUseFlag();
    size_t nxtInUse = nxtChunk.inUseFlag();

    if (preInUse && nxtInUse) // no need to coalesce
        return chunk.mem();
    else if (preInUse && !nxtInUse) { // coalesce the next chunk
        chunk.set(chunk.size() + nxtSize, 0);
        return chunk.mem();
    } else if (!preInUse && nxtInUse) { // coalesce the prev chunk
        preChunk.set(chunk.size() + preSize, 0);
        return preChunk.mem();
    } else { // coalesce the prev and nxt chunke
        preChunk.set(chunk.size() + preSize + nxtSize, 0);
        return preChunk.mem();
    }
}

void Bin::free(void *p) {
    Chunk victim(p);
    victim.set(victim.size(), 0); // don't modify the size field
    coalesce(victim.mem());
}

void *Bin::malloc(SIZE_T size) {
    char *p;
    SIZE_T asize;       // Adjusted chunk size
    SIZE_T extendsize;  // Amount to extend heap if no fit
    /*Ignore spurious request*/
    if (size == 0)
        return nullptr;

    /*Adjust block size to include overhead and alignment reqs*/
    if (size <= DSIZE) // minimal size
        asize = 2 * DSIZE;
    else // align to 2 * DSIZE
        asize = (size + DSIZE - 1) & ~(DSIZE - 1);

    /*Search the free_list for a fit*/
    if ((p = find_fit(asize)) != nullptr) {
        place(p, asize);
        return p;
    }

    /*No fit found. Get more memory and place the block*/
    extendsize = asize > CHUNKSIZE ? asize : CHUNKSIZE;
    if ((p = extend_heap(extendsize)) == nullptr) {
        return nullptr;
    }

    place(p, asize);
    return p;
}

char *Bin::find_fit(SIZE_T asize) {
    auto p = Chunk(free_list);
    while (p.size() > 0) {
        // WARNING: this is different from book. But I consider that I was right.
        if (p.size() - DSIZE >= asize && p.inUseFlag() == 0) {
            return p.mem();
        }
        p = p.nxtChunk();
    }

    return (p.size() > 0 ? p.mem() : nullptr);
}

/*
 * f == footer, h == header, r == remainder
 *           h       p                                       f             nxtChunk
 *           | WSIZE |                    p->size()                           |
 *           | WSIZE |      allocated_size         |      remainder_size      |
 *           [ WSIZE |    asize    | WSIZE | WSIZE |         | WSIZE  | WSIZE ]
 *           h       p             f       h       r         f        h    nxtChunk
 */

void Bin::place(char *p, SIZE_T asize) {
    Chunk chunk(p);
    size_t sz = chunk.size();
    if (sz - asize >= 2 * DSIZE) {
        SIZE_T allocated_size = asize + DSIZE;
        SIZE_T remainder_size = sz - allocated_size;

        chunk.set(allocated_size, 1); // allocate
        Chunk remainder((char *) p + allocated_size);
        remainder.set(remainder_size, 0);
    } else {
        chunk.set(sz, 1);
    }
}

void Chunk::checkChunk() {
    auto address = reinterpret_cast<unsigned long>(&(*mem()));
//  std::cout<<std::hex<<address<<std::endl;
    if (address % 8) {
        std::cerr << "Error: " << std::hex << address << " is not double word aligned\n";
    }
    if (get(header()) != get(footer()))
        std::cerr << "Error: header does not match footer\n";
}


void Bin::checkHeap(int verbose) {
    if (verbose)
        std::cout << "Heap: " << static_cast<void *>(free_list) << std::endl;

    Chunk p(free_list);
    if (p.size() != DSIZE || !p.inUseFlag())
        std::cerr << "Bad prologue header\n";
    p.checkChunk();

    for (p = p.nxtChunk(); p.size() > 0; p = p.nxtChunk()) {
        if (verbose)
            p.print();
        p.checkChunk();
    }

    if (verbose) // epilogue don't have footer, so the value we don't care
        p.print();
    if (p.size() != 0 || p.inUseFlag() == 0)
        std::cerr << "Bad epilogue header\n";
}


#ifdef MMMALLOC_DEBUG

#include "iostream"

void Chunk::print() {

    std::cout << "header:\t\t\t";
    std::cout << static_cast<void *>(header()) << std::endl;

    std::cout << "mem:\t\t\t";
    std::cout << static_cast<void *>(m_mem) << std::endl;

    std::cout << "footer:\t\t\t";
    std::cout << static_cast<void *>(footer()) << std::endl;

    std::cout << "size:\t\t\t";
    std::cout << size() << ", " << (*footer() & ~0x7) << std::endl;

    std::cout << "bit:\t\t\t";
    std::cout << inUseFlag() << ", " << (*footer() & 0x1) << std::endl;

    std::cout << "Next chunk:\t\t";
    std::cout << static_cast<void *>(nxtChunk().mem()) << std::endl;

    std::cout << "Prev chunk:\t\t";
    std::cout << static_cast<void *>(preChunk().mem()) << std::endl;

    std::cout << std::endl;
}
#endif


