#define FMT_HEADER_ONLY

#include <fmt/color.h>
#include <fmt/format.h>
#include <mysql.h>
#include <iostream>

using namespace std;

int main(int argc, const char** argv) {
  MYSQL* mysql = mysql_init(nullptr);
  mysql_real_connect(mysql, "localhost", "root", "passwd", "test", 3306,
                     nullptr, 0);

  int limit;

  limit = argc >= 2 ? atoi(argv[1]) : 10;
  for (int i = 0; i < limit; i++) {
    string data = fmt::format(
        "INSERT INTO student(name,sex,age,city) Values({0}, {1}, {2}, {3});",
        "'maojingfang'", "'female'", 31, "'shanghai'");
    fmt::print(fmt::emphasis::bold | fg(fmt::color::green_yellow),
               "insert {}\n", i + 1);
    if (mysql_query(mysql, data.c_str()) != 0)
      fmt::print(fg(fmt::color::dark_red), "{}", "Error in insert\n");
  }
  mysql_close(mysql);
  return 0;
}
