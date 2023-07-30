#include <benchmark/benchmark.h>
#include <string>

void Func1(std::string_view str) {}

void Func2(const std::string& str) {}

static void UseStringView(benchmark::State& state) {
  for (auto _ : state) {  // NOLINT
    // Use string_view: no need to allocate memory buffer
    Func1("Hello world");
  }
}

BENCHMARK(UseStringView);  // NOLINT

static void UseString(benchmark::State& state) {
  for (auto _ : state) {  // NOLINT
    // Use string: we must allocate memory to save string.
    // If the string is very long, this may be a huge time consuming
    Func2("Hello world");
  }
}

BENCHMARK(UseString);  // NOLINT

static void UseLongStringView(benchmark::State& state) {
  for (auto _ : state) {  // NOLINT
    // Use string_view: no need to allocate memory buffer
    Func1("Hello world Hello world Hello world Hello world Hello world ");
  }
}

BENCHMARK(UseLongStringView);  // NOLINT

static void UseLongString(benchmark::State& state) {
  for (auto _ : state) {  // NOLINT
    // Use string: we must allocate memory to save string.
    // If the string is very long, this may be a huge time consuming
    Func2("Hello world Hello world Hello world Hello world Hello world ");
  }
}

BENCHMARK(UseLongString);  // NOLINT

BENCHMARK_MAIN();  // NOLINT
