#pragma once

#include "ops.hpp"
#include "tensor.hpp"

namespace tinytorch {

struct SGDOptimizer  // stochastic gradient descent
{
  int iter = 0;
  float lr;
  float momentum = 0.9;
  std::vector<Tensor> params;

  SGDOptimizer(std::vector<Tensor> t_lst, float lr) : params(t_lst), lr(lr) {}

  void zeroGrad() {
    for (Tensor& t : params) {
      t.clearGrad();
    }
  }

  void step() {
    // update the weight of params
    for (size_t p = 0; p < params.size(); p++) {
      Tensor& param = params[p];
      if (param.grad().size() == 0) {
        continue;
      }
      for (size_t i = 0; i < param.size(); i++) {
        auto& w = param[i];
        assert(param.grad().size() == param.size());
        float g = param.grad()[i];

        // sgd with nesterov momentum, equation from pytorch
        if (iter > 0) {
          float b = momentum * b + 0.9 * g;
          g = g + momentum * b;
        }

        w = w - lr * g;

      }
    }
    iter++;
  }
};

}  // namespace tinytorch
