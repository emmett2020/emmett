#include <iostream>

#include <cstddef>
#include <stdexcept>
#include <type_traits>

template <typename... Types>
struct Variant {
  static constexpr std::size_t InvalidIndex = static_cast<std::size_t>(-1);

  // 计算最大对齐和最大尺寸
  static constexpr std::size_t max_size() {
    std::size_t max = 0;
    ((sizeof(Types) > max ? max = sizeof(Types) : 0), ...);
    return max;
  }

  static constexpr std::size_t max_align() {
    std::size_t max = 0;
    ((alignof(Types) > max ? max = alignof(Types) : 0), ...);
    return max;
  }

  // 存储数据的缓冲区
  alignas(max_align()) char storage_[max_size()]{};
  std::size_t index_ = InvalidIndex;

  // 类型索引映射
  template <typename T>
  static constexpr std::size_t index_of() {
    std::size_t idx = 0;
    bool found      = false;
    ((found = found || std::is_same_v<T, Types> ? true : (idx++, false)), ...);
    return found ? idx : InvalidIndex;
  }

  // 根据索引销毁对象
  void destroy() {
    if (index_ == InvalidIndex) {
      return;
    }
    [this]<std::size_t... Is>(std::index_sequence<Is...>) {
      ((Is == index_
          ? (reinterpret_cast<std::tuple_element_t<Is, std::tuple<Types...>>*>(storage_)->~Types(),
             void())
          : void()),
       ...);
    }(std::make_index_sequence<sizeof...(Types)>{});
    index_ = InvalidIndex;
  }

public:
  // 构造/析构
  Variant() = default;

  template <typename T, typename = std::enable_if_t<(index_of<T>() != InvalidIndex)>>
  explicit Variant(T&& value) {
    emplace<T>(std::forward<T>(value));
  }

  ~Variant() {
    destroy();
  }

  // 赋值操作
  template <typename T>
  Variant& operator=(T&& value) {
    destroy();
    emplace<T>(std::forward<T>(value));
    return *this;
  }

  // 原位构造
  template <typename T, typename... Args>
  void emplace(Args&&... args) {
    static_assert(index_of<T>() != InvalidIndex, "Type not allowed");
    destroy();
    new (storage_) T(std::forward<Args>(args)...);
    index_ = index_of<T>();
  }

  // 获取当前索引
  [[nodiscard]] std::size_t index() const {
    return index_;
  }

  // 获取值
  template <std::size_t I>
  auto& get() {
    using T = std::tuple_element_t<I, std::tuple<Types...>>;
    if (index_ != I) {
      throw std::runtime_error("Bad variant access");
    }
    return *reinterpret_cast<T*>(storage_);
  }

  template <typename T>
  T& get() {
    constexpr std::size_t idx = index_of<T>();
    if (index_ != idx) {
      std::cout << index_ << "\n";
      throw std::runtime_error("Bad variant access");
    }
    return *reinterpret_cast<T*>(storage_);
  }
};

int main() {
  Variant<int, double, string> v;
  v.emplace<string>(std::string("123"));
  // std::cout << v.get<string>() << "\n";

  std::cout << Variant<int, double, int, float>::max_size();
}

