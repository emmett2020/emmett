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

#include <iostream>
#include <memory>

struct abstract_product {
  abstract_product()                                           = default;
  abstract_product(const abstract_product&)                    = default;
  abstract_product(abstract_product&&)                         = default;
  auto operator=(const abstract_product&) -> abstract_product& = default;
  auto operator=(abstract_product&&) -> abstract_product&      = default;

  virtual ~abstract_product() = default;
  virtual void use()          = 0;
};

struct product_a : public abstract_product {
  void use() override {
    std::cout << "use product A\n";
  }
};

struct product_b : public abstract_product {
  void use() override {
    std::cout << "use product B\n";
  }
};

struct abstract_factory {
  abstract_factory()                                                 = default;
  abstract_factory(const abstract_factory&)                          = default;
  abstract_factory(abstract_factory&&)                               = default;
  auto operator=(const abstract_factory&) -> abstract_factory&       = default;
  auto operator=(abstract_factory&&) -> abstract_factory&            = default;
  virtual ~abstract_factory()                                        = default;
  virtual auto create_product() -> std::shared_ptr<abstract_product> = 0;
};

struct factory_a : abstract_factory {
  auto create_product() -> std::shared_ptr<abstract_product> override {
    return std::make_shared<product_a>();
  }
};

struct factory_b : abstract_factory {
  auto create_product() -> std::shared_ptr<abstract_product> override {
    return std::make_shared<product_b>();
  }
};

auto main() -> int {
  auto fact_a    = factory_a{};
  auto fact_b    = factory_b{};
  auto product_a = fact_a.create_product();
  auto product_b = fact_b.create_product();
  product_a->use();
  product_b->use();
}

