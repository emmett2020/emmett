#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <csignal>
#include <iostream>

using namespace std;

void handler(int sig) {
  cout << "capture signal: " << sys_siglist[sig] << endl;
}

int main() {
  signal(SIGBUS, handler);
  int fd = open("../test_sigbus.txt", O_RDWR);
  char* addr = (char*)mmap(nullptr, 0, PROT_WRITE, MAP_FILE, fd, 0);
  getchar();
  write(fd, "hi ", 4);
  return 0;
}