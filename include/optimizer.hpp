#pragma once

#include "ops.hpp"
#include "tensor.hpp"

namespace microtorch {

struct SGDOptimizer  // stochastic gradient descent
{
  int iter = 0;
  float lr;
  float momentum;
  float dampening;
  std::vector<Tensor> params;
  std::vector<Tensor> velocities;

  SGDOptimizer(std::vector<Tensor> t_lst, float lr, float momentum = 0.0,
               float dampening = 0.0)
      : params(t_lst), lr(lr), momentum(momentum), dampening(dampening) {
    // initialize velocities
    velocities.resize(t_lst.size());
    for (size_t i = 0; i < t_lst.size(); ++i) {
      velocities[i] = zeros(t_lst[i].shape(), t_lst[i].device());
    }
  }

  void zeroGrad() {
    for (Tensor &t : params) {
      if (t.grad().defined()) {
        t.grad().zero_();
      }
    }
  }

  void step() {
    // update the weight of params
    // sgd with nesterov momentum, equation from pytorch
    // see https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    for (size_t p = 0; p < params.size(); p++) {
      Tensor &param = params[p];
      if (!param.grad().defined()) {
        continue;
      }
      TORCH_CHECK(param.grad().numel() == param.numel(),
                  "grad size and size should be equal.");
      Tensor grad = param.grad();
      Tensor update_value;
      if (momentum != 0) {
        if (iter > 0) {
          velocities[p] = velocities[p] * momentum + grad * (1 - dampening);
          if (dampening != 0) {
            update_value = grad + velocities[p] * momentum;
          } else {
            update_value = velocities[p];
          }
        } else {
          velocities[p] = grad.clone();
          update_value = grad;
        }
      } else {
        update_value = grad;
      }
      param -= update_value * lr;
    }
    iter++;
  }
};

}  // namespace microtorch
