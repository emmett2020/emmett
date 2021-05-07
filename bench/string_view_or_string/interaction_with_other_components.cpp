#include <array>
#include <benchmark/benchmark.h>
#include <string>

/*
  --------------------------------------------------------
  Benchmark              Time             CPU   Iterations
  --------------------------------------------------------
  UseStringView      0.000 ns        0.000 ns   1000000000000
  UseString          0.000 ns        0.000 ns   1000000000000
 */

// Use other components as parameter
// Func1({arr.data(), 1024});         // array<char, 1024> arr;
// Func1({vec.data(), vec.size()});   // vector<char> vec;
// Func1({c_style, 1024});            // char c_style[1024];
void Func1(std::string_view str) {}

void Func2(const std::string &str) {}

static void UseStringView(benchmark::State &state) {
  std::array<char, 1024> str{"hello world"};
  for (auto _ : state) {
    Func1({str.data(), str.size()});
  }
}

BENCHMARK(UseStringView); // NOLINT

static void UseString(benchmark::State &state) {
  std::array<char, 1024> str{"hello world"};
  for (auto _ : state) {
    Func1({str.begin(), str.end()});
  }
}

BENCHMARK(UseString); // NOLINT

BENCHMARK_MAIN(); // NOLINT
