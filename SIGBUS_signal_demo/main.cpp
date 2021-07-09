#include <iostream>
#include <unistd.h>
#include <csignal>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>



using namespace std;

void handler(int sig) {

    cout << "capture signal: " << sys_siglist[sig] << endl;
}

int main(){
//    mmap()

    signal(SIGBUS,handler);

    int fd = open("../test_sigbus.txt",O_RDWR);
    char* addr = (char*) mmap(nullptr,0,PROT_WRITE,MAP_FILE,fd,0);
    getchar();

    write(fd,"hi ",4);

    return 0;
}