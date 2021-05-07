#include <csignal>
#include <fcntl.h>
#include <print>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

using namespace std; // NOLINT

// TODO: FIX sys_siglist not found
void handler(int sig) { std::println("capture signal: ", sys_siglist[sig]); }

int main() {
  ::signal(SIGBUS, handler);
  int fd = open("../test_sigbus.txt", O_RDWR | O_CLOEXEC);
  auto *addr =
      static_cast<char *>(mmap(nullptr, 0, PROT_WRITE, MAP_FILE, fd, 0));
  getchar();
  ::write(fd, "hi ", 4);
  return 0;
}
