#define FMT_HEADER_ONLY
#include <iostream>
#include <fmt/format.h>
#include "demo_mmmalloc/heap.h"

using namespace  std;
int main(){
    T1 t1;
    fmt::print("hello,{}","world");
    return 0;
}