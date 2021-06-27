cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 20)
project(add_data_to_mysql)

INCLUDE_DIRECTORIES(
        include
        /usr/local/mysql/include # mysql 头文件
        /opt/homebrew/Cellar/fmt/7.1.3/include
)

LINK_DIRECTORIES(
        /usr/local/mysql/lib
        /opt/homebrew/Cellar/fmt/7.1.3/lib
        )

LINK_LIBRARIES(libmysqlclient.a)



add_executable(add_data_to_mysql main.cpp)

target_link_libraries(add_data_to_mysql libmysqlclient.a)