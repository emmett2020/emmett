message(STATUS "Building tools: add_data_to_mysql")

INCLUDE_DIRECTORIES(
        include
        /usr/local/mysql/include # mysql
        /opt/homebrew/Cellar/fmt/7.1.3/include
)

LINK_DIRECTORIES(
        /usr/local/mysql/lib
        /opt/homebrew/Cellar/fmt/7.1.3/lib
        )

LINK_LIBRARIES(libmysqlclient.a)

add_executable(add_data_to_mysql add_data_to_mysql.cpp)
target_link_libraries(add_data_to_mysql libmysqlclient.a)
