#include <iostream>
#include "demo_mmmalloc/mm.h"
using namespace std;

int main(){
    Bin b;
    auto p = b.malloc(10);
    b.checkHeap(1);
    b.free(p);
    cout<<endl<<endl;
    b.checkHeap(1);



    return 0;
}
