/**
 * Copyright (c) 2022-2024 yewentao256
 * Licensed under the MIT License.
 */
#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

namespace microtorch {

// An efficient, type-erasing, non-owning reference to a callable.

// Note: FuncRef does not own the lifecycle of the callable it references.
// As such, users must ensure:
// 1. The referenced callable (e.g., function, lambda, functor) remains alive
//    and in scope for as long as the function_ref instance might be invoked.
// 2. Be cautious when passing temporary callables to function_ref, as they
//    might be destroyed at the end of the expression they were created in.

template <typename Fn>
class FuncRef;

template <typename Ret, typename... Params>
class FuncRef<Ret(Params...)> {
 private:
  // `callback` member variable, a func pointer to `callable_fn` template
  Ret (*callback)(intptr_t callable, Params... params) = nullptr;
  // `callable` member variable, storing the address of `Callable`
  intptr_t callable;

  template <typename Callable>
  static Ret callback_fn(intptr_t callable_fn, Params... params) {
    return (*reinterpret_cast<Callable*>(callable_fn))(
        std::forward<Params>(params)...);
  }

 public:
  FuncRef() = default;
  FuncRef(std::nullptr_t) {}

  // SFINAE (Substitution Failure Is Not An Error)
  // Callable&& callable: right valued referrence, accepting any callable
  // Valid if Callable is not a FuncRef and can return a `Ret` value
  template <typename Callable>
  FuncRef(Callable&& callable,
          std::enable_if_t<
              !std::is_same_v<std::remove_reference_t<Callable>, FuncRef> &&
                  std::is_convertible_v<
                      std::invoke_result_t<Callable, Params...>, Ret>,
              int> = 0)
      : callback(callback_fn<typename std::remove_reference<Callable>::type>),
        callable(reinterpret_cast<intptr_t>(&callable)) {}

  Ret operator()(Params... params) const {
    // using `callback` to call `callable`
    return callback(callable, std::forward<Params>(params)...);
  }

  operator bool() const { return callback; }
};

}  // namespace microtorch