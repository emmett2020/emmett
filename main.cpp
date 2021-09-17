#include <iostream>
#include "demo_mmmalloc/mm.h"
using namespace std;

int main(){
    Bin b;

    for(int i = 0; i < 1000; ++i) {
        auto p = (char *) b.malloc(8);
        cout<<reinterpret_cast<long>(p)<<endl;

        auto p2 = (char *) b.malloc(16);
        cout<<reinterpret_cast<long>(p2)<<endl;

        auto p3 = (char *) b.malloc(24);
        cout<<reinterpret_cast<long>(p3)<<endl;
    }



    return 0;
}
