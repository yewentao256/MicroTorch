/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#pragma once

#include <tuple>

// Modified from
// https://stackoverflow.com/questions/7943525/is-it-possible-to-figure-out-the-parameter-type-and-return-type-of-a-lambda

namespace microtorch {

// Main template, fetching address of T's operator(), decltype
// and then call specific template specialization.
template <typename T>
struct FuncTraits : public FuncTraits<decltype(&T::operator())> {};

// For class member function pointer
template <typename ClassType, typename T>
struct FuncTraits<T ClassType::*> : public FuncTraits<T> {};

// For const class member functions
template <typename ClassType, typename ReturnType, typename... Args>
struct FuncTraits<ReturnType (ClassType::*)(Args...) const>
    : public FuncTraits<ReturnType(Args...)> {};

// Reference types
template <typename T>
struct FuncTraits<T&> : public FuncTraits<T> {};
template <typename T>
struct FuncTraits<T*> : public FuncTraits<T> {};

// Free functions
template <typename ReturnType, typename... Args>
struct FuncTraits<ReturnType(Args...)> {
  // arity is the number of arguments.
  enum { arity = sizeof...(Args) };

  typedef std::tuple<Args...> ArgsTuple;
  typedef ReturnType result_type;

  template <size_t i>
  struct arg {
    typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
    // the i-th argument is equivalent to the i-th tuple element of a tuple
    // composed of those arguments.
  };
};

}  // namespace microtorch