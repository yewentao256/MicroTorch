/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#pragma once

#include "tensor.hpp"

namespace microtorch {

// stochastic gradient descent optimizer
struct SGDOptimizer
{
  int iter_ = 0;
  float lr_;
  float momentum_;
  float dampening_;
  std::vector<Tensor> params_;
  std::vector<Tensor> velocities_;

  SGDOptimizer(std::vector<Tensor> t_lst, float lr, float momentum = 0.0,
               float dampening = 0.0);

  // clear the grad
  void zeroGrad();

  // step to next iter, calculating the gradients
  void step();
};

}  // namespace microtorch
