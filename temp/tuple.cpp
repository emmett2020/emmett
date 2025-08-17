#include <iostream>
#include <type_traits>
#include <utility>
#include <string>

// 前置声明
template <typename... Types>
class Tuple;

// 递归基类：空Tuple
template <>
class Tuple<> { };

// 递归定义：非空Tuple
template <typename Head, typename... Tail>
class Tuple<Head, Tail...> : private Tuple<Tail...> {
  Head value_;

public:
  // 构造函数
  Tuple() = default;

  Tuple(const Head& head, const Tail&... tail)
    : Tuple<Tail...>(tail...)
    , value_(head) {
  }

  Tuple(Head&& head, Tail&&... tail)
    : Tuple<Tail...>(std::forward<Tail>(tail)...)
    , value_(std::forward<Head>(head)) {
  }

  // 获取元素（通过索引）
  template <std::size_t I>
  auto& get() {
    if constexpr (I == 0) {
      return value_;
    } else {
      return Tuple<Tail...>::template get<I - 1>();
    }
  }

  // 获取元素（通过类型）
  template <typename T>
  auto& get() {
    if constexpr (std::is_same_v<T, Head>) {
      return value_;
    } else {
      return Tuple<Tail...>::template get<T>();
    }
  }
};

// ---------------------------------------------------
// 辅助函数：make_tuple（类似std::make_tuple）
template <typename... Types>
auto make_tuple(Types&&... args) {
  return Tuple<std::decay_t<Types>...>(std::forward<Types>(args)...);
}

// ---------------------------------------------------
// 示例用法：
int main() {
  // 创建Tuple
  Tuple<int, float, std::string> t(42, 3.14f, "hello");

  // 通过索引访问
  auto& i = t.get<0>(); // int 42
  auto& f = t.get<1>(); // float 3.14
  auto& s = t.get<2>(); // string "hello"

  // 通过类型访问（类型必须唯一）
  auto& i2 = t.get<int>();         // 42
  auto& s2 = t.get<std::string>(); // "hello"

  // 使用make_tuple
  auto t2 = make_tuple(1, 2.0, "three");

  std::cout << i << f << s << "\n";
  return 0;
}
