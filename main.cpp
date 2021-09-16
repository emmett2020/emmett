#define FMT_HEADER_ONLY
#include <iostream>
#include <fmt/format.h>
using namespace  std;
int main(){
    fmt::print("hello,{}","world");
    return 0;
}