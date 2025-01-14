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
#include <string>

#include <benchmark/benchmark.h>

/*
--------------------------------------------------------------
Benchmark                    Time             CPU   Iterations
--------------------------------------------------------------
UseStringView            0.000 ns        0.000 ns   1000000000000
UseLongStringView        0.000 ns        0.000 ns   1000000000000
UseString                0.000 ns        0.000 ns   1000000000000
UseLongString             9.05 ns         9.05 ns     66258755
UseCStyleString          0.000 ns        0.000 ns   1000000000000
UseLongCStyleString      0.000 ns        0.000 ns   1000000000000
*/

void Func(std::string_view str) {}

static void UseStringView(benchmark::State &state) {
  for (auto _ : state) {
    // Use string_view: no need to allocate memory buffer
    Func("Hello world");
  }
}

BENCHMARK(UseStringView);

static void UseLongStringView(benchmark::State &state) {
  for (auto _ : state) {
    // Use string_view: no need to allocate memory buffer
    Func("Hello world Hello world Hello world Hello world Hello world ");
  }
}

BENCHMARK(UseLongStringView);

void Func2(const std::string &str) {}
static void UseString(benchmark::State &state) {
  for (auto _ : state) {
    // Use string: we must allocate memory to save string.
    // If the string is very long, this may be a huge time consuming
    Func2("Hello world");
  }
}

BENCHMARK(UseString);

static void UseLongString(benchmark::State &state) {
  for (auto _ : state) {
    // Use string: we must allocate memory to save string.
    // If the string is very long, this may be a huge time consuming
    Func2("Hello world Hello world Hello world Hello world Hello world ");
  }
}

BENCHMARK(UseLongString);

void Func3(const char *str) {}

static void UseCStyleString(benchmark::State &state) {
  for (auto _ : state) {
    Func3("Hello world");
  }
}

BENCHMARK(UseCStyleString);

static void UseLongCStyleString(benchmark::State &state) {
  for (auto _ : state) {
    Func3("Hello world Hello world Hello world Hello world Hello world ");
  }
}

BENCHMARK(UseLongCStyleString);

BENCHMARK_MAIN();
