#pragma once

#include <mutex>

namespace tinytorch {

template <typename Tag, typename Func>
void log_once(Tag t, Func f){
    static std::once_flag flag;
    std::call_once(flag, f);
} 

}  // namespace tinytorch