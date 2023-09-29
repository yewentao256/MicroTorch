#pragma once
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
  oss << value;
  printTemplate(oss, args...);
}

#define TORCH_CHECK(cond, ...)                                          \
  do {                                                                  \
    if (!(cond)) {                                                      \
      std::ostringstream oss;                                           \
      oss << "Assertion failed: " << #cond << " at " << __FILE__ << ":" \
          << __LINE__ << "\n";                                          \
      oss << "Additional Information: ";                                \
      printTemplate(oss, __VA_ARGS__);                                  \
      throw std::runtime_error(oss.str());                              \
    }                                                                   \
  } while (false)

}  // namespace microtorch
