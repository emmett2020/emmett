
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

struct AbstractFactory {
  AbstractFactory() = default;
  AbstractFactory(const AbstractFactory&) = default;
  AbstractFactory(AbstractFactory&&) = default;
  auto operator=(const AbstractFactory&) -> AbstractFactory& = default;
  auto operator=(AbstractFactory&&) -> AbstractFactory& = default;
  virtual ~AbstractFactory() = default;
  virtual auto createProduct() -> std::shared_ptr<AbstractProduct> = 0;
};

struct FactoryA : AbstractFactory {
  auto createProduct() -> std::shared_ptr<AbstractProduct> override {
    return std::make_shared<ProductA>();
  }
};

struct FactoryB : AbstractFactory {
  auto createProduct() -> std::shared_ptr<AbstractProduct> override {
    return std::make_shared<ProductB>();
  }
};

auto main() -> int {
  FactoryA factory_a;
  FactoryB factory_b;
  auto product_a = factory_a.createProduct();
  auto product_b = factory_b.createProduct();
  product_a->use();
  product_b->use();
}