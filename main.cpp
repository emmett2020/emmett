#include <iostream>
#include "demo_mmmalloc/mm.h"
using namespace std;

int main(){
    Bin b;
    auto p = (char*)b.malloc(8);
    auto q1 = (int*)p;
    auto q2 = (int*)p + 2;
    *q1 = 3;
    *q2 = 4;
    cout<<*q1<<" "<<*q2<<endl;
    b.free(p);
    return 0;
}
