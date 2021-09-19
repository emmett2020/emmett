#include <iostream>
#include "demo_gcmalloc/gcmalloc.h"
using namespace std;

int main(){
    GC gc;
    auto p = gc.malloc(10);
    



    return 0;
}
