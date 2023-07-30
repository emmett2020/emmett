#include <benchmark/benchmark.h>
#include <string>

void Func1(std::string_view str) {}

void Func2(const std::string& str) {}

void Func3(const char* str) {}

static void UseStringView(benchmark::State& state) {
  for (auto _ : state) {
    // Use string_view: no need to allocate memory buffer
    Func1("Hello world");
  }
}

BENCHMARK(UseStringView);  // NOLINT

static void UseString(benchmark::State& state) {
  for (auto _ : state) {
    // Use string: we must allocate memory to save string.
    // If the string is very long, this may be a huge time consuming
    Func2("Hello world");
  }
}

BENCHMARK(UseString);  // NOLINT

static void UseLongStringView(benchmark::State& state) {
  for (auto _ : state) {
    // Use string_view: no need to allocate memory buffer
    Func1("Hello world Hello world Hello world Hello world Hello world ");
  }
}

BENCHMARK(UseLongStringView);  // NOLINT

static void UseLongString(benchmark::State& state) {
  for (auto _ : state) {
    // Use string: we must allocate memory to save string.
    // If the string is very long, this may be a huge time consuming
    Func2("Hello world Hello world Hello world Hello world Hello world ");
  }
}

BENCHMARK(UseLongString);  // NOLINT

static void UseCStyleString(benchmark::State& state) {
  for (auto _ : state) {
    Func3("Hello world");
  }
}

BENCHMARK(UseCStyleString);  // NOLINT

static void UseLongCStyleString(benchmark::State& state) {
  for (auto _ : state) {
    Func3("Hello world Hello world Hello world Hello world Hello world ");
  }
}

BENCHMARK(UseLongCStyleString);  // NOLINT

BENCHMARK_MAIN();  // NOLINT
