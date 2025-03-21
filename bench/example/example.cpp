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
#include <cstring>

#include <benchmark/benchmark.h>

static void Func1(benchmark::State &state) {
  char c = '8';
  for (auto _: state) {
    c += 1;
  }
}

BENCHMARK(Func1);

static void Func2(benchmark::State &state) {
  char c = '8';
  for (auto _: state) {
    c -= 1;
  }
}

BENCHMARK(Func2);

BENCHMARK_MAIN();

