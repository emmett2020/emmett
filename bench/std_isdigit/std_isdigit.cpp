#include <benchmark/benchmark.h>
#include <string>

// The fastest way is use operator directly. Otherwise we can lookup the table.
// The slowest way is to use std::is_xxx.

static constexpr std::array<uint8_t, 256> kTokenTable{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //   0
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //  16
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //  32
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,  //  48
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //  64
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //  80
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //  96
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //  112
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //  128
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //  144
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //  160
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //  176
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //  192
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //  208
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //  224
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0   //  240
};

static constexpr std::array<uint8_t, 256> kAlphaTable{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //   0
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //  16
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //  32
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //  48
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  //  64
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,  //  80
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  //  96
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,  //  112
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //  128
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //  144
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //  160
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //  176
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //  192
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //  208
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  //  224
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0   //  240
};

inline constexpr bool IsAlpha(uint8_t input) noexcept {
  return static_cast<bool>(kAlphaTable[input]);  // NOLINT
}

inline constexpr bool IsDigit(uint8_t input) noexcept {
  return static_cast<bool>(kTokenTable[input]);  // NOLINT
}

static void UseStdIsDigit(benchmark::State& state) {
  char c = '8';
  for (auto _ : state) {
    if (std::isdigit(c) == 0) {}
  }
}

BENCHMARK(UseStdIsDigit);  // NOLINT

static void UsIsDigit(benchmark::State& state) {
  char c = '8';
  for (auto _ : state) {
    if (IsDigit(c)) {}
  }
}

BENCHMARK(UsIsDigit);  // NOLINT

static void UsConstexprIsDigit(benchmark::State& state) {
  for (auto _ : state) {
    if (IsDigit('8')) {}
  }
}

BENCHMARK(UsConstexprIsDigit);  // NOLINT

static void UsOperator(benchmark::State& state) {
  char c = '8';
  for (auto _ : state) {
    if (c >= '0' && c <= '9') {}
  }
}

BENCHMARK(UsOperator);  // NOLINT

static void UseStdIsAlpha(benchmark::State& state) {
  char c = 'd';
  for (auto _ : state) {
    if (std::isalpha(c) == 0) {}
  }
}

BENCHMARK(UseStdIsAlpha);  // NOLINT

static void UsIsAlpha(benchmark::State& state) {
  char c = 'd';
  for (auto _ : state) {
    if (IsAlpha(c)) {}
  }
}

BENCHMARK(UsIsAlpha);  // NOLINT

static void UsConstexprIsAlpha(benchmark::State& state) {
  for (auto _ : state) {
    if (IsAlpha('d')) {}
  }
}

BENCHMARK(UsConstexprIsAlpha);  // NOLINT

static void UsOperator2(benchmark::State& state) {
  char c = 'd';
  for (auto _ : state) {
    if ((c | 0x20) >= 'a' && (c | 0x20) <= 'z') {}
  }
}

BENCHMARK(UsOperator2);  // NOLINT

BENCHMARK_MAIN();  // NOLINT
