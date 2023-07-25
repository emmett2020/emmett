#include <iostream>
#include <memory>

struct AbstractProduct {
  AbstractProduct() = default;
  AbstractProduct(const AbstractProduct&) = default;
  AbstractProduct(AbstractProduct&&) = default;
  auto operator=(const AbstractProduct&) -> AbstractProduct& = default;
  auto operator=(AbstractProduct&&) -> AbstractProduct& = default;

  virtual ~AbstractProduct() = default;
  virtual void use() = 0;
};

struct ProductA : public AbstractProduct {
  void use() override { std::cout << "use product A\n"; }
};

struct ProductB : public AbstractProduct {
  void use() override { std::cout << "use product B\n"; }
};

struct Factory {
  static auto createProduct(const std::string& name)
      -> std::shared_ptr<AbstractProduct> {
    if (name == "A") {
      return std::make_shared<ProductA>();
    }
    if (name == "B") {
      return std::make_shared<ProductB>();
    }
    return nullptr;
  }
};

auto main() -> int {
  auto product_a = Factory::createProduct("A");
  auto product_b = Factory::createProduct("B");
  product_a->use();
  product_b->use();
}