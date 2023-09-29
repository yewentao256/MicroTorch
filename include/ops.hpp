#pragma once

#include "context.hpp"
#include "tensor.hpp"
#include "tensorFactories.hpp"
#include "binaryOps.hpp"
#include "unaryOps.hpp"
#include "reduceOps.hpp"

namespace microtorch {

std::string print_with_size(Tensor t, int64_t print_size = 20,
                 const std::string& name = "name");
std::ostream& operator<<(std::ostream& stream, Tensor t);

void initialize_ops();
}  // namespace microtorch
