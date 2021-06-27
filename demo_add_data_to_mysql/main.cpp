//
// Created by 张乃港 on 2021/6/27.
//
#define FMT_HEADER_ONLY

#include <mysql.h>
#include <fmt/format.h>
#include <fmt/color.h>
#include <iostream>

using namespace std;

int main(int argc, const char **argv) {
    MYSQL *mysql = mysql_init(nullptr);
    mysql_real_connect(mysql, "localhost", "root", "passwd", "test", 3306, nullptr, 0);

    int limit;

    limit = argc >= 2 ? atoi(argv[1]) : 10;
    // 添加10万条数据
    for (int i = 0; i < limit; i++) {
        string data = fmt::format(
                "INSERT INTO student(name,sex,age,city) Values({0}, {1}, {2}, {3});",
                "'maojingfang'", "'female'", 31, "'shanghai'");
        fmt::print(fmt::emphasis::bold | fg(fmt::color::green_yellow), "正在插入第{}条数据\n", i + 1);
        if (mysql_query(mysql, data.c_str()) != 0)
            fmt::print(fg(fmt::color::dark_red), "{}", "Error in insert\n");
    }

//    mysql_query(mysql,"SELECT * from student;");
//    MYSQL_RES* mysqlRes = mysql_store_result(mysql);
//
//    unsigned num_fileds;
//    unsigned i;
//    MYSQL_FIELD *fields;
//    MYSQL_ROW row;
//
//    num_fileds = mysql_num_fields(mysqlRes);
//    fields = mysql_fetch_field(mysqlRes);
//
//    for (i = 0; i < num_fileds; i++)
//    {
//        fmt::print(fmt::emphasis::bold |  fg(fmt::color::red),"{:20}",fields[i].name);
//    }
//    cout<<endl;
//
//    while((row = mysql_fetch_row(mysqlRes)) != nullptr)
//    {
//        for (i = 0; i < num_fileds; i++) {
//            fmt::print(fmt::emphasis::bold |  fg(fmt::color::green_yellow),"{:20}",row[i]);
//        }
//        std::cout<<std::endl;
//    }
//
//    mysql_free_result(mysqlRes);
    mysql_close(mysql);
    return 0;
}


