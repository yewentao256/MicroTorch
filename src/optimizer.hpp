#pragma once

#include "ops.hpp"
#include "tensor.hpp"

namespace tinytorch {

struct SGDOptimizer  // stochastic gradient descent
{
  int iter = 0;
  float momentum = 0.9;
  float dampening = 0.1;
  std::vector<Tensor> params;
  std::vector<Tensor> velocities;
  float lr;
  bool nesterov = true;

  SGDOptimizer(std::vector<Tensor> t_lst, float lr) : params(t_lst), lr(lr) {
    // initialize velocities
    velocities.resize(t_lst.size());
    for (size_t i = 0; i < t_lst.size(); ++i) {
      velocities[i] = zero(t_lst[i].size());
    }
  }

  void zeroGrad() {
    for (Tensor &t : params) {
      t.clearGrad();
    }
  }

  void step() {
    // update the weight of params
    for (size_t p = 0; p < params.size(); p++) {
      Tensor &param = params[p];
      Tensor &velocity = velocities[p];
      if (param.grad().size() == 0) {
        continue;
      }
      for (size_t i = 0; i < param.size(); i++) {
        assert(param.grad().size() == param.size());
        auto &w = param[i];
        auto g = param.grad()[i];
        auto &b = velocity[i];
        // sgd with nesterov momentum, equation from pytorch
        // see https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
        if (momentum != 0) {
          if (iter > 0) {
            b = momentum * b + (1 - dampening) * g;
          } else {
            b = g;
          }
          if (nesterov) {
            g = g + momentum * b;
          } else {
            g = b;
          }
        }
        w = w - lr * g;
      }
    }
    iter++;
  }
};

}  // namespace tinytorch
