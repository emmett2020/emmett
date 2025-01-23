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

struct factory {
  static auto create_product(const std::string& name) -> std::shared_ptr<abstract_product> {
    if (name == "A") {
      return std::make_shared<product_a>();
    }
    if (name == "B") {
      return std::make_shared<product_b>();
    }
    return nullptr;
  }
};

auto main() -> int {
  auto product_a = factory::create_product("A");
  auto product_b = factory::create_product("B");
  product_a->use();
  product_b->use();
}


