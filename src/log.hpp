#pragma once

#include <mutex>

namespace tinytorch {

template <typename Tag, typename Func>
void log_once(Tag t, Func f) {
  static std::once_flag flag;
  std::call_once(flag, f);
}

/* log_once([]() {},                                                      \
        []() {                                                        \
            std::cout << "LOG: using" + ctx.arch + "func `" #func "` "  \
                    << std::endl;                                     \
        });                                                           \ */

}  // namespace tinytorch