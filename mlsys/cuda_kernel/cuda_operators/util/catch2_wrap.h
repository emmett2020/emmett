#pragma once

#include <catch2/catch_all.hpp>

/// We make this wrap since libtorch also defined "CHECK" macro. We have
/// no other choice but to add prefix 'CATCH_' to catch2 macros. However we don't want
/// to use such long name of all macros, so we manunally define short-style macros back.

#define REQUIRE(...)                                                                  \
  INTERNAL_CATCH_TEST("CATCH_REQUIRE", Catch::ResultDisposition::Normal, __VA_ARGS__)
#define REQUIRE_FALSE(...)                                                                    \
  INTERNAL_CATCH_TEST("CATCH_REQUIRE_FALSE",                                                  \
                      Catch::ResultDisposition::Normal | Catch::ResultDisposition::FalseTest, \
                      __VA_ARGS__)

#define REQUIRE_THROWS(...)                                                                    \
  INTERNAL_CATCH_THROWS("CATCH_REQUIRE_THROWS", Catch::ResultDisposition::Normal, __VA_ARGS__)
#define REQUIRE_THROWS_AS(expr, exceptionType) \
  INTERNAL_CATCH_THROWS_AS(                    \
    "CATCH_REQUIRE_THROWS_AS",                 \
    exceptionType,                             \
    Catch::ResultDisposition::Normal,          \
    expr)
#define REQUIRE_NOTHROW(...)                                                                      \
  INTERNAL_CATCH_NO_THROW("CATCH_REQUIRE_NOTHROW", Catch::ResultDisposition::Normal, __VA_ARGS__)

#define CHECK_FALSE(...)                                                               \
  INTERNAL_CATCH_TEST(                                                                 \
    "CATCH_CHECK_FALSE",                                                               \
    Catch::ResultDisposition::ContinueOnFailure | Catch::ResultDisposition::FalseTest, \
    __VA_ARGS__)
#define CHECKED_IF(...)                                                                   \
  INTERNAL_CATCH_IF(                                                                      \
    "CATCH_CHECKED_IF",                                                                   \
    Catch::ResultDisposition::ContinueOnFailure | Catch::ResultDisposition::SuppressFail, \
    __VA_ARGS__)
#define CHECKED_ELSE(...)                                                                 \
  INTERNAL_CATCH_ELSE(                                                                    \
    "CATCH_CHECKED_ELSE",                                                                 \
    Catch::ResultDisposition::ContinueOnFailure | Catch::ResultDisposition::SuppressFail, \
    __VA_ARGS__)
#define CHECK_NOFAIL(...)                                                                 \
  INTERNAL_CATCH_TEST(                                                                    \
    "CATCH_CHECK_NOFAIL",                                                                 \
    Catch::ResultDisposition::ContinueOnFailure | Catch::ResultDisposition::SuppressFail, \
    __VA_ARGS__)

#define CHECK_THROWS(...)                                            \
  INTERNAL_CATCH_THROWS("CATCH_CHECK_THROWS",                        \
                        Catch::ResultDisposition::ContinueOnFailure, \
                        __VA_ARGS__)
#define CHECK_THROWS_AS(expr, exceptionType)     \
  INTERNAL_CATCH_THROWS_AS(                      \
    "CATCH_CHECK_THROWS_AS",                     \
    exceptionType,                               \
    Catch::ResultDisposition::ContinueOnFailure, \
    expr)
#define CHECK_NOTHROW(...)                                             \
  INTERNAL_CATCH_NO_THROW("CATCH_CHECK_NOTHROW",                       \
                          Catch::ResultDisposition::ContinueOnFailure, \
                          __VA_ARGS__)

#define TEST_CASE(...)                   INTERNAL_CATCH_TESTCASE(__VA_ARGS__)
#define TEST_CASE_METHOD(className, ...) INTERNAL_CATCH_TEST_CASE_METHOD(className, __VA_ARGS__)
#define METHOD_AS_TEST_CASE(method, ...) INTERNAL_CATCH_METHOD_AS_TEST_CASE(method, __VA_ARGS__)
#define TEST_CASE_PERSISTENT_FIXTURE(className, ...)                  \
  INTERNAL_CATCH_TEST_CASE_PERSISTENT_FIXTURE(className, __VA_ARGS__)
#define REGISTER_TEST_CASE(Function, ...) INTERNAL_CATCH_REGISTER_TESTCASE(Function, __VA_ARGS__)
#define SECTION(...)                      INTERNAL_CATCH_SECTION(__VA_ARGS__)
#define DYNAMIC_SECTION(...)              INTERNAL_CATCH_DYNAMIC_SECTION(__VA_ARGS__)
#define FAIL(...)                      \
  INTERNAL_CATCH_MSG(                  \
    "CATCH_FAIL",                      \
    Catch::ResultWas::ExplicitFailure, \
    Catch::ResultDisposition::Normal,  \
    __VA_ARGS__)
#define FAIL_CHECK(...)                          \
  INTERNAL_CATCH_MSG(                            \
    "CATCH_FAIL_CHECK",                          \
    Catch::ResultWas::ExplicitFailure,           \
    Catch::ResultDisposition::ContinueOnFailure, \
    __VA_ARGS__)
#define SUCCEED(...)                             \
  INTERNAL_CATCH_MSG(                            \
    "CATCH_SUCCEED",                             \
    Catch::ResultWas::Ok,                        \
    Catch::ResultDisposition::ContinueOnFailure, \
    __VA_ARGS__)
#define SKIP(...)                     \
  INTERNAL_CATCH_MSG(                 \
    "SKIP",                           \
    Catch::ResultWas::ExplicitSkip,   \
    Catch::ResultDisposition::Normal, \
    __VA_ARGS__)


