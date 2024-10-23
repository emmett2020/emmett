#include <cstring>

#include <benchmark/benchmark.h>

// Provide a way to tell if a character is
//  a digit
//  or a alpha
//  or in a collection of digit, alpha and other special characters.

// Result:
// The fastest way is to use operator directly. But it make the code not easy to
// read. Especially when the expression becomes complicated.

// The awesome way is to lookup the table and elide the function call use always
// inline attribute.

//  The slowest way is to use std::is_xxx. One shouldn't use it in a base
//  library.

// Real output with gcc:

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
