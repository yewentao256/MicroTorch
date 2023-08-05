#pragma once

#include "context.hpp"
#include "tensor.hpp"
#include "tensorFactories.hpp"
#include "binaryOps.hpp"
#include "unaryOps.hpp"
#include "reduceOps.hpp"

namespace tinytorch {

std::string print_with_size(Tensor t, size_t print_size = 20,
                 const std::string& name = "name");
std::ostream& operator<<(std::ostream& stream, Tensor t);

void initialize_ops();
}  // namespace tinytorch
