#include <benchmark/benchmark.h>
#include <string>

/*
  --------------------------------------------------------------
  Benchmark                    Time             CPU   Iterations
  --------------------------------------------------------------
  UseStringView            0.000 ns        0.000 ns   1000000000000
  UseString                0.000 ns        0.000 ns   1000000000000
  UseLongStringView        0.000 ns        0.000 ns   1000000000000
  UseLongString             17.0 ns         17.0 ns     41341635
  UseCStyleString          0.000 ns        0.000 ns   1000000000000
  UseLongCStyleString      0.000 ns        0.000 ns   1000000000000
*/

void Func1(std::string_view str) {}

void Func2(const std::string &str) {}

void Func3(const char *str) {}

static void UseStringView(benchmark::State &state) {
  for (auto _ : state) {
    // Use string_view: no need to allocate memory buffer
    Func1("Hello world");
  }
}

BENCHMARK(UseStringView); // NOLINT

static void UseString(benchmark::State &state) {
  for (auto _ : state) {
    // Use string: we must allocate memory to save string.
    // If the string is very long, this may be a huge time consuming
    Func2("Hello world");
  }
}

BENCHMARK(UseString); // NOLINT

static void UseLongStringView(benchmark::State &state) {
  for (auto _ : state) {
    // Use string_view: no need to allocate memory buffer
    Func1("Hello world Hello world Hello world Hello world Hello world ");
  }
}

BENCHMARK(UseLongStringView); // NOLINT

static void UseLongString(benchmark::State &state) {
  for (auto _ : state) {
    // Use string: we must allocate memory to save string.
    // If the string is very long, this may be a huge time consuming
    Func2("Hello world Hello world Hello world Hello world Hello world ");
  }
}

BENCHMARK(UseLongString); // NOLINT

static void UseCStyleString(benchmark::State &state) {
  for (auto _ : state) {
    Func3("Hello world");
  }
}

BENCHMARK(UseCStyleString); // NOLINT

static void UseLongCStyleString(benchmark::State &state) {
  for (auto _ : state) {
    Func3("Hello world Hello world Hello world Hello world Hello world ");
  }
}

BENCHMARK(UseLongCStyleString); // NOLINT

BENCHMARK_MAIN(); // NOLINT
