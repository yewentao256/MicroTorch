/**
 * Copyright (c) 2022-2024 yewentao256
 * Licensed under the MIT License.
 */
#pragma once

#include "context.hpp"
#include "tensor.hpp"
#include "tensorFactories.hpp"
#include "binaryOps.hpp"
#include "unaryOps.hpp"
#include "reduceOps.hpp"

namespace microtorch {

std::string print_with_size(const Tensor t, int64_t print_size = 20,
                 const std::string& name = "name");
std::ostream& operator<<(std::ostream& stream, const Tensor t);

}  // namespace microtorch
