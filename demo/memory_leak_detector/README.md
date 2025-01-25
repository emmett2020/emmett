## Catalogue
1. [Memory Management in C++: Tools to Prevent Leaks](#Memory Management in C++: Tools to Prevent Leaks)
2. [第二章](#第二章)
3. [第三章](#第三章)

# Memory Management in C++: Tools to Prevent Leaks
Memory management in C++ is a critical yet challenging aspect of development, especially when manually allocating and deallocating memory. Relying solely on developers to avoid memory leaks is not a reliable approach. To address this, tools like [valgrind](https://valgrind.org/info/tools.html#memcheck) and [AddressSanitizer (ASan)](https://github.com/google/sanitizers/wiki/AddressSanitizer) are invaluable for detecting and preventing memory-related issues.

Depp dive into valgrind:
![valgrind](./docs/valgrind_design.png)

Below is a detailed comparison of the key differences between and ASan:

| **Feature**         | **Valgrind**                             | **AddressSanitizer (ASan)**             |
|-----------------|--------------------------------------|-------------------------------------|
| **Type**            | Dynamic analysis tool                | Compiler-integrated tool            |
| **Performance**     | Slower (runs in a virtual machine)   | Faster (compiled into the program)  |
| **Ease of Use**     | Requires no code changes             | Requires recompilation with flags   |
| **Detection Scope** | Broad (memory leaks, invalid access) | Focused (memory errors, leaks)      |
| **Platform Support**| Cross-platform (Linux, macOS, etc.)  | Primarily Linux, macOS, and Windows |


By leveraging these tools, developers can significantly reduce the risk of memory leaks and improve the overall stability of their applications.

# Run this demo with valgrind:
```bash
cmake -S . -B build -GNinja
cmake --build build
valgrind --leak-check=full ./build/memory_leak
valgrind --leak-check=full ./build/safely_allocate_memory
```


# Run this demo with ASAN:
```bash
# Just add some compilation arguments to your CMakeLists.txt
set(CMAKE_CXX_FLAGS "-fsanitize=address -fno-omit-frame-pointer -g -O1 ${CMAKE_CXX_FLAGS}")

# -fsanitize=address: Enables AddressSanitizer.
# -fno-omit-frame-pointer: Ensures stack traces are readable.
# -g: Includes debugging symbols for better error reporting.
# -O1: get better running performance

cmake -S . -B build -GNinja
cmake --build build
./build/memory_leak
./build/safely_allocate_memory
```

If everything fine, you'll get:
```text
=================================================================
==69839==ERROR: AddressSanitizer: alloc-dealloc-mismatch (operator new [] vs operator delete) on 0x525000000100
    #0 0xffff904e9444 in operator delete(void*, unsigned long) ../../../../src/libsanitizer/asan/asan_new_delete.cpp:164
    #1 0xaaaad38a08ec in allocate_free /emmett/demo/memory_leak_detector/safely_allocate_memory.cpp:6
    #2 0xaaaad38a08ec in main /emmett/demo/memory_leak_detector/safely_allocate_memory.cpp:11
    #3 0xffff8fe684c0 in __libc_start_call_main ../sysdeps/nptl/libc_start_call_main.h:58
    #4 0xffff8fe68594 in __libc_start_main_impl ../csu/libc-start.c:360
    #5 0xaaaad38a07ec in _start (/emmett/demo/memory_leak_detector/build/safely_allocate_memory+0x7ec) (BuildId: 4e9e174d78055b1a9763b90667fb98f606a4967f)

0x525000000100 is located 0 bytes inside of 8192-byte region [0x525000000100,0x525000002100)
allocated by thread T0 here:
    #0 0xffff904e84a8 in operator new[](unsigned long) ../../../../src/libsanitizer/asan/asan_new_delete.cpp:98
    #1 0xaaaad38a08e4 in allocate_free /emmett/demo/memory_leak_detector/safely_allocate_memory.cpp:4
    #2 0xaaaad38a08e4 in main /emmett/demo/memory_leak_detector/safely_allocate_memory.cpp:11
    #3 0xffff8fe684c0 in __libc_start_call_main ../sysdeps/nptl/libc_start_call_main.h:58
    #4 0xffff8fe68594 in __libc_start_main_impl ../csu/libc-start.c:360
    #5 0xaaaad38a07ec in _start (/emmett/demo/memory_leak_detector/build/safely_allocate_memory+0x7ec) (BuildId: 4e9e174d78055b1a9763b90667fb98f606a4967f)

SUMMARY: AddressSanitizer: alloc-dealloc-mismatch ../../../../src/libsanitizer/asan/asan_new_delete.cpp:164 in operator delete(void*, unsigned long)
==69839==HINT: if you don't care about these errors you may set ASAN_OPTIONS=alloc_dealloc_mismatch=0
==69839==ABORTING
```
