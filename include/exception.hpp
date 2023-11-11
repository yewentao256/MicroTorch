/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#pragma once
#include <execinfo.h>
#include <unistd.h>

#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace microtorch {

template <typename T>
void printTemplate(std::ostringstream& oss, const T& value) {
  oss << value << " ";
}

template <typename T, typename... Args>
void printTemplate(std::ostringstream& oss, const T& value,
                   const Args&... args) {
  oss << value << " ";
  printTemplate(oss, args...);
}

constexpr int MaxStackFrames = 128;

inline void printStackTrace(std::ostringstream& oss, int skip = 1, int maxFramesToPrint = 10) {
  void* callstack[MaxStackFrames];
  int nFrames = backtrace(callstack, MaxStackFrames);
  char** symbols = backtrace_symbols(callstack, nFrames);

  oss << "Stack Trace:\n";
  // Print up to maxFramesToPrint frames, starting from 'skip'
  for (int i = skip; i < nFrames && i < skip + maxFramesToPrint; i++) {
    oss << symbols[i] << "\n";
  }

  free(symbols);
}

#define TORCH_CHECK(cond, ...)                                          \
  do {                                                                  \
    if (!(cond)) {                                                      \
      std::ostringstream oss;                                           \
      printStackTrace(oss, 1);                                          \
      oss << "Assertion failed: " << #cond << " at " << __FILE__ << ":" \
          << __LINE__ << " in " << __PRETTY_FUNCTION__ << "\n";         \
      oss << "Additional Information: ";                                \
      printTemplate(oss, __VA_ARGS__);                                  \
      throw std::runtime_error(oss.str());                              \
    }                                                                   \
  } while (false)

#define TORCH_INTERNAL_ASSERT(cond)                                           \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::ostringstream oss;                                                 \
      printStackTrace(oss, 1);                                                \
      oss << "INTERNAL ASSERT FAILED: " << #cond << " at " << __FILE__ << ":" \
          << __LINE__ << " in " << __PRETTY_FUNCTION__ << "\n";               \
      oss << "Please report a bug to MicroTorch.\n";                          \
      throw std::runtime_error(oss.str());                                    \
    }                                                                         \
  } while (false)

}  // namespace microtorch
