/*
 * Copyright (c) 2024 Emmett Zhang
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <array>
#include <cstdint>
#include <cstring>

#include <benchmark/benchmark.h>

// Provide a way to tell if a character is a digit or a alpha or other special
// characters.

// Debug mode:
/*
------------------------------------------------------------------------
Benchmark                              Time             CPU   Iterations
------------------------------------------------------------------------
UseStdIsDigit                       4.76 ns         4.76 ns    146559106
UseIsDigit                          3.87 ns         3.87 ns    180774853
UseAlwaysInlineIsDigit              2.99 ns         2.98 ns    234886717
UseConstexprIsDigit                 2.30 ns         2.30 ns    303399373
UseSimpleOperator                   2.68 ns         2.68 ns    261538519
UseStdIsAlpha                       5.36 ns         5.35 ns    130782453
UseIsAlpha                          3.94 ns         3.94 ns    180693191
UseAlwaysInlineIsAlpha              3.01 ns         3.01 ns    229035857
UseConstexprIsAlpha                 2.37 ns         2.36 ns    303934211
UseSomehowComplicatedOperator       2.79 ns         2.74 ns    261406667
UseIsToken                          3.92 ns         3.92 ns    175761045
UseAlwaysInlineIsToken              3.08 ns         3.07 ns    227173892
UseComplicatedOperator              3.07 ns         2.97 ns    231476889
*/

// Release mode:

/*
------------------------------------------------------------------------
Benchmark                              Time             CPU   Iterations
------------------------------------------------------------------------
UseStdIsDigit                      0.000 ns        0.000 ns   1000000000000
UseIsDigit                         0.000 ns        0.000 ns   1000000000000
UseAlwaysInlineIsDigit             0.000 ns        0.000 ns   1000000000000
UseConstexprIsDigit                0.000 ns        0.000 ns   1000000000000
UseSimpleOperator                  0.000 ns        0.000 ns   1000000000000
UseStdIsAlpha                      0.000 ns        0.000 ns   1000000000000
UseIsAlpha                         0.000 ns        0.000 ns   1000000000000
UseAlwaysInlineIsAlpha             0.000 ns        0.000 ns   1000000000000
UseConstexprIsAlpha                0.000 ns        0.000 ns   1000000000000
UseSomehowComplicatedOperator      0.000 ns        0.000 ns   1000000000000
UseIsToken                         0.000 ns        0.000 ns   1000000000000
UseAlwaysInlineIsToken             0.000 ns        0.000 ns   1000000000000
UseComplicatedOperator             0.000 ns        0.000 ns   1000000000000
 */

static constexpr std::array<uint8_t, 256> kDigitTable{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //   0
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //  16
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //  32
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, //  48
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //  64
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //  80
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //  96
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //  112
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //  128
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //  144
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //  160
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //  176
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //  192
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //  208
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //  224
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  //  240
};

static constexpr std::array<uint8_t, 256> kAlphaTable{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //   0
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //  16
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //  32
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //  48
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, //  64
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, //  80
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, //  96
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, //  112
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //  128
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //  144
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //  160
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //  176
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //  192
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //  208
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //  224
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  //  240
};

constexpr std::array<uint8_t, 256> kTokenTable = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 0
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 16
    0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, // 32
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, // 48
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 64
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, // 80
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 96
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, // 112
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 128
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 144
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 160
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 176
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 192
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 208
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 224
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 240-255
};

constexpr bool IsAlpha(uint8_t input) noexcept {
  return static_cast<bool>(kAlphaTable[input]);
}

constexpr bool IsDigit(uint8_t input) noexcept {
  return static_cast<bool>(kDigitTable[input]);
}

constexpr bool IsToken(uint8_t input) noexcept {
  return static_cast<bool>(kTokenTable[input]);
}

__attribute__((always_inline)) constexpr bool
AlwaysInlineIsAlpha(uint8_t input) noexcept {
  return static_cast<bool>(kAlphaTable[input]);
}

__attribute__((always_inline)) constexpr bool
AlwaysInlineIsDigit(uint8_t input) noexcept {
  return static_cast<bool>(kDigitTable[input]);
}

__attribute__((always_inline)) constexpr bool
ALwaysInlineIsToken(uint8_t input) noexcept {
  return static_cast<bool>(kTokenTable[input]);
}

static void UseStdIsDigit(benchmark::State &state) {
  char c = '8';
  for (auto _ : state) {
    if (std::isdigit(c) == 0) {
    }
  }
}

BENCHMARK(UseStdIsDigit);

static void UseIsDigit(benchmark::State &state) {
  char c = '8';
  for (auto _ : state) {
    if (IsDigit(c)) {
    }
  }
}

BENCHMARK(UseIsDigit);

static void UseAlwaysInlineIsDigit(benchmark::State &state) {
  char c = '8';
  for (auto _ : state) {
    if (AlwaysInlineIsDigit(c)) {
    }
  }
}

BENCHMARK(UseAlwaysInlineIsDigit);

static void UseConstexprIsDigit(benchmark::State &state) {
  for (auto _ : state) {
    if (IsDigit('8')) {
    }
  }
}

BENCHMARK(UseConstexprIsDigit);

static void UseSimpleOperator(benchmark::State &state) {
  char c = '8';
  for (auto _ : state) {
    if (c >= '0' && c <= '9') {
    }
  }
}

BENCHMARK(UseSimpleOperator);

static void UseStdIsAlpha(benchmark::State &state) {
  char c = 'd';
  for (auto _ : state) {
    if (std::isalpha(c) == 0) {
    }
  }
}

BENCHMARK(UseStdIsAlpha);

static void UseIsAlpha(benchmark::State &state) {
  char c = 'd';
  for (auto _ : state) {
    if (IsAlpha(c)) {
    }
  }
}

BENCHMARK(UseIsAlpha);

static void UseAlwaysInlineIsAlpha(benchmark::State &state) {
  char c = 'd';
  for (auto _ : state) {
    if (AlwaysInlineIsAlpha(c)) {
    }
  }
}

BENCHMARK(UseAlwaysInlineIsAlpha);

static void UseConstexprIsAlpha(benchmark::State &state) {
  for (auto _ : state) {
    if (IsAlpha('d')) {
    }
  }
}

BENCHMARK(UseConstexprIsAlpha);

static void UseSomehowComplicatedOperator(benchmark::State &state) {
  char c = 'd';
  for (auto _ : state) {
    if ((c | 0x20) >= 'a' && (c | 0x20) <= 'z') {
    }
  }
}

BENCHMARK(UseSomehowComplicatedOperator);

static void UseIsToken(benchmark::State &state) {
  char c = 'd';
  for (auto _ : state) {
    if (IsToken(c)) {
    }
  }
}

BENCHMARK(UseIsToken);

static void UseAlwaysInlineIsToken(benchmark::State &state) {
  char c = 'd';
  for (auto _ : state) {
    if (ALwaysInlineIsToken(c)) {
    }
  }
}

BENCHMARK(UseAlwaysInlineIsToken);

static void UseComplicatedOperator(benchmark::State &state) {
  char c = 'd';
  for (auto _ : state) {
    if (((c | 0x20) >= 'a' && (c | 0x20) <= 'z') || c == 33 ||
        (c >= 35 && c <= 39) || (c >= 42 && c <= 46 && c != 44)) {
    }
  }
}

BENCHMARK(UseComplicatedOperator);

BENCHMARK_MAIN();
