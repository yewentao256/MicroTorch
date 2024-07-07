/**
 * Copyright (c) 2022-2024 yewentao256
 * Licensed under the MIT License.
 */
#include "optimizer.hpp"

#include "graph.hpp"
#include "ops.hpp"

namespace microtorch {

SGDOptimizer::SGDOptimizer(std::vector<Tensor> t_lst, float lr, float momentum,
                           float dampening)
    : lr_(lr), momentum_(momentum), dampening_(dampening), params_(t_lst) {
  // initialize velocities
  velocities_.resize(t_lst.size());
  for (size_t i = 0; i < t_lst.size(); ++i) {
    velocities_[i] = zeros(t_lst[i].shape(), t_lst[i].device());
  }
}

void SGDOptimizer::zeroGrad() {
  for (Tensor &t : params_) {
    if (t.grad().defined()) {
      t.grad().zero_();
    }
  }
}

void SGDOptimizer::step() {
  // update the weight of params
  // sgd with nesterov momentum, equation from pytorch
  AutoGradGuard guard(false);
  for (size_t p = 0; p < params_.size(); p++) {
    Tensor &param = params_[p];
    if (!param.grad().defined()) {
      continue;
    }
    TORCH_CHECK(param.grad().numel() == param.numel(),
                "grad size and size should be equal.");
    Tensor grad = param.grad();
    Tensor update_value;
    if (momentum_ != 0) {
      if (iter_ > 0) {
        velocities_[p] = velocities_[p] * momentum_ + grad * (1 - dampening_);
        if (dampening_ != 0) {
          update_value = grad + velocities_[p] * momentum_;
        } else {
          update_value = velocities_[p];
        }
      } else {
        velocities_[p] = grad.clone();
        update_value = grad;
      }
    } else {
      update_value = grad;
    }
    param -= update_value * lr_;
  }
  iter_++;
}

}  // namespace microtorch
