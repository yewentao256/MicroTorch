/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#pragma once

#include <tuple>

// Modified from
// https://stackoverflow.com/questions/7943525/is-it-possible-to-figure-out-the-parameter-type-and-return-type-of-a-lambda

namespace microtorch {

// Handle lambda function object
// eg: FuncTraits<decltype(lambdaFunction)> ->
// FuncTraits<decltype(&TestClass::constMemberFunction)> ->
// FuncTraits<decltype(normalFunction)>
// Why calling for constMemberFunction?
// Because lambda is an instance of an anonymous class containing `operator()`
// member function. The constness of this `operator()` depends on whether the
// lambda is declared as mutable. eg: auto mutableLambda = [&]() mutable { };
template <typename T>
struct FuncTraits : public FuncTraits<decltype(&T::operator())> {
};

// Handle class member function pointer
// eg: FuncTraits<decltype(&TestClass::memberFunction)> ->
// FuncTraits<decltype(normalFunction)>
template <typename ClassType, typename T>
struct FuncTraits<T ClassType::*> : public FuncTraits<T> {
};

// Handle const class member functions
// eg: FuncTraits<decltype(&TestClass::constMemberFunction)> ->
// FuncTraits<decltype(normalFunction)>
template <typename ClassType, typename ResultType, typename... Args>
struct FuncTraits<ResultType (ClassType::*)(Args...) const>
    : public FuncTraits<ResultType(Args...)> {
};

// Handle function reference
// eg: FuncTraits<decltype((void (&)(int))normalFunction)> ->
// FuncTraits<decltype(normalFunction)>;
template <typename T>
struct FuncTraits<T&> : public FuncTraits<T> {
};

// Handle function pointer
// eg: FuncTraits<decltype(&normalFunction)> ->
// FuncTraits<decltype(normalFunction)>;
template <typename T>
struct FuncTraits<T*> : public FuncTraits<T> {
};

// Main base template to handle normal functions
// eg: FuncTraits<decltype(normalFunction)>;
template <typename ResultType, typename... Args>
struct FuncTraits<ResultType(Args...)> {
  typedef std::tuple<Args...> ArgsTuple;
  typedef ResultType result_type;

  static constexpr size_t num_args = sizeof...(Args);
  template <size_t i>
  struct arg {
    // usage: FuncTraits<..>::arg<i>::type
    // fetching the element through i in tuple
    typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
  };
};

}  // namespace microtorch