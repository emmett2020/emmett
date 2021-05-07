#include <cstring>

#include <benchmark/benchmark.h>

static void Func1(benchmark::State &state) {
  char c = '8';
  for (auto _ : state) {
    c += 1;
  }
}

BENCHMARK(Func1); // NOLINT

static void Func2(benchmark::State &state) {
  char c = '8';
  for (auto _ : state) {
    c -= 1;
  }
}

BENCHMARK(Func2); // NOLINT

BENCHMARK_MAIN(); // NOLINT
