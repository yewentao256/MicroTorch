/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#include "ops.hpp"

#include "engine.hpp"
#include "graph.hpp"

namespace microtorch {

std::ostream& print_with_size(std::ostream& stream, const Tensor t,
                              int64_t print_size,
                              const std::string& name = "name") {
  if (!t.defined()) {
    stream << "<microtorch.Tensor[ undefined ]>";
    return stream;
  }
  int64_t size = t.numel();
  TORCH_CHECK(t.shape().size() == 1, "only support 1d tensor print now");
  stream << "<microtorch.Tensor[" << name << "] size=" << size
         << ", device=" << t.device() << ", requires_grad=" << t.requires_grad()
         << ", storage_ptr: " << t.data_ptr() << ">: [";
  if (size > print_size) {
    for (int64_t i = 0; i < print_size / 2; i++) {
      stream << std::setw(8) << t[i] << " ";
    }
    stream << " ... ";
    for (int64_t i = size - print_size / 2; i < size; i++) {
      stream << std::setw(8) << t[i] << " ";
    }
  } else {
    for (int64_t i = 0; i < size; i++) {
      stream << std::setw(8) << t[i] << " ";
    }
  }
  stream << "]";
  return stream;
}

std::string print_with_size(const Tensor t, int64_t print_size,
                            const std::string& name) {
  std::ostringstream s;
  print_with_size(s, t, print_size, name);
  return s.str();
}

std::ostream& operator<<(std::ostream& stream, const Tensor t) {
  return print_with_size(stream, t, 20);
}

struct AddNode : public FunctionNode<AddNode> {
  static std::vector<Tensor> forward(Context& ctx, const Tensor& a,
                                     const Tensor& b) {
    ctx.data.emplace("a", a);
    ctx.data.emplace("b", b);
    Tensor out;
    DISPATCH_OP(add_impl, a.device(), a, b, out);
    return {out};
  }
  static std::vector<Tensor> backward(Context& ctx,
                                      std::vector<Tensor>& grads) {
    auto& grad_output = grads[0];
    Tensor grad_input_1 =
        zeros(ctx.data.at("a").shape(), ctx.data.at("a").device());
    Tensor grad_input_2 =
        zeros(ctx.data.at("b").shape(), ctx.data.at("b").device());
    DISPATCH_OP(add_backward_impl, ctx.device, grad_output, grad_input_1,
                grad_input_2);
    return {grad_input_1, grad_input_2};
  }
};

Tensor Tensor::operator+(const Tensor& other) {
  check_device_shape({*this, other});
  return AddNode::forward_and_build_graph(*this, other)[0];
}

Tensor& Tensor::operator+=(const Tensor& other) {
  check_device_shape({*this, other});
  TORCH_CHECK(
      !this->requires_grad() || !GradModeController::is_enabled(),
      "Tensor that requires grad is being used in an in-place operation");
  DISPATCH_OP(add_impl, this->device(), *this, other, *this);
  return *this;
}

struct SubNode : public FunctionNode<SubNode> {
  static std::vector<Tensor> forward(Context& ctx, const Tensor& a,
                                     const Tensor& b) {
    Tensor out = zeros(a.shape(), a.device(), a.requires_grad());
    DISPATCH_OP(sub_impl, a.device(), a, b, out);
    return {out};
  }
  static std::vector<Tensor> backward(Context& ctx,
                                      std::vector<Tensor>& grads) {
    auto& grad_output = grads[0];
    Tensor grad_input_1 = zeros(grad_output.shape(), grad_output.device());
    Tensor grad_input_2 = zeros(grad_output.shape(), grad_output.device());
    DISPATCH_OP(sub_backward_impl, ctx.device, grad_output, grad_input_1,
                grad_input_2);
    return {grad_input_1, grad_input_2};
  }
};

Tensor Tensor::operator-(const Tensor& other) {
  check_device_shape({*this, other});
  return SubNode::forward_and_build_graph(*this, other)[0];
}

Tensor& Tensor::operator-=(const Tensor& other) {
  check_device_shape({*this, other});
  TORCH_CHECK(
      !this->requires_grad() || !GradModeController::is_enabled(),
      "Tensor that requires grad is being used in an in-place operation");
  DISPATCH_OP(sub_impl, this->device(), *this, other, *this);
  return *this;
}

struct MulNode : public FunctionNode<MulNode> {
  static std::vector<Tensor> forward(Context& ctx, const Tensor& a,
                                     const Tensor& b) {
    // save tensor data to context
    ctx.data.emplace("a", a);
    ctx.data.emplace("b", b);
    Tensor out = zeros(a.shape(), a.device(), a.requires_grad());
    DISPATCH_OP(mul_impl, a.device(), a, b, out);
    return {out};
  }

  static std::vector<Tensor> backward(Context& ctx,
                                      std::vector<Tensor>& grads) {
    auto& grad_output = grads[0];
    Tensor grad_input_1 = zeros(grad_output.shape(), grad_output.device());
    Tensor grad_input_2 = zeros(grad_output.shape(), grad_output.device());
    DISPATCH_OP(mul_backward_impl, ctx.device, grad_output, grad_input_1,
                grad_input_2, ctx.data.at("a"), ctx.data.at("b"));
    return {grad_input_1, grad_input_2};
  }
};

Tensor Tensor::operator*(const Tensor& other) {
  check_device_shape({*this, other});
  return MulNode::forward_and_build_graph(*this, other)[0];
}
Tensor& Tensor::operator*=(const Tensor& other) {
  check_device_shape({*this, other});
  TORCH_CHECK(
      !this->requires_grad() || !GradModeController::is_enabled(),
      "Tensor that requires grad is being used in an in-place operation");
  DISPATCH_OP(mul_impl, this->device(), *this, other, *this);
  return *this;
}

struct MulScalarNode : public FunctionNode<MulScalarNode> {
  static std::vector<Tensor> forward(Context& ctx, const Tensor& a,
                                     const data_t b) {
    ctx.data_scalar.emplace("b", b);
    Tensor out = zeros(a.shape(), a.device(), a.requires_grad());
    DISPATCH_OP(mul_scalar_impl, a.device(), a, b, out);
    return {out};
  }

  static std::vector<Tensor> backward(Context& ctx,
                                      std::vector<Tensor>& grads) {
    auto& grad_output = grads[0];
    Tensor grad_input = zeros(grad_output.shape(), grad_output.device());
    DISPATCH_OP(mul_scalar_backward_impl, ctx.device, grad_output, grad_input,
                ctx.data_scalar.at("b"));
    return {grad_input};
  }
};

Tensor Tensor::operator*(const data_t other) {
  return MulScalarNode::forward_and_build_graph(*this, other)[0];
}

Tensor& Tensor::operator*=(const data_t other) {
  TORCH_CHECK(
      !this->requires_grad() || !GradModeController::is_enabled(),
      "Tensor that requires grad is being used in an in-place operation");
  DISPATCH_OP(mul_scalar_impl, this->device(), *this, other, *this);
  return *this;
}

struct DivNode : public FunctionNode<DivNode> {
  static std::vector<Tensor> forward(Context& ctx, const Tensor& a,
                                     const Tensor& b) {
    // save tensor data to context
    ctx.data.emplace("a", a);
    ctx.data.emplace("b", b);
    Tensor out = zeros(a.shape(), a.device(), a.requires_grad());
    DISPATCH_OP(div_impl, a.device(), a, b, out);
    return {out};
  }

  static std::vector<Tensor> backward(Context& ctx,
                                      std::vector<Tensor>& grads) {
    auto& grad_output = grads[0];
    Tensor grad_input_1 = zeros(grad_output.shape(), grad_output.device());
    Tensor grad_input_2 = zeros(grad_output.shape(), grad_output.device());
    DISPATCH_OP(div_backward_impl, ctx.device, grad_output, grad_input_1,
                grad_input_2, ctx.data.at("a"), ctx.data.at("b"));
    return {grad_input_1, grad_input_2};
  }
};

Tensor Tensor::operator/(const Tensor& other) {
  check_device_shape({*this, other});
  return DivNode::forward_and_build_graph(*this, other)[0];
}
Tensor& Tensor::operator/=(const Tensor& other) {
  check_device_shape({*this, other});
  TORCH_CHECK(
      !this->requires_grad() || !GradModeController::is_enabled(),
      "Tensor that requires grad is being used in an in-place operation");
  DISPATCH_OP(div_impl, this->device(), *this, other, *this);
  return *this;
}

struct SumNode : public FunctionNode<SumNode> {
  static std::vector<Tensor> forward(Context& ctx, const Tensor& input) {
    // save tensor data to context
    ctx.data.emplace("input", input);
    Tensor out = zeros({1}, input.device(), input.requires_grad());
    DISPATCH_OP(sum_impl, input.device(), input, out);
    return {out};
  }

  static std::vector<Tensor> backward(Context& ctx,
                                      std::vector<Tensor>& grads) {
    Tensor& input = ctx.data.at("input");
    const Tensor& grad_output = grads[0];
    TORCH_CHECK(grad_output.numel() == 1,
                "grad_output numel should equal to 1");
    // y = a + b + c ..., y'(a) = 1 * grad[0], y'(b) = 1 * grad[1] ...
    Tensor grad_input = zeros(input.shape(), input.device());
    DISPATCH_OP(fill_impl, ctx.device, grad_input, grad_output[0]);
    return {grad_input};
  }
};

Tensor sum(const Tensor& a) { return SumNode::forward_and_build_graph(a)[0]; }

struct SumDimNode : public FunctionNode<SumDimNode> {
  static std::vector<Tensor> forward(Context& ctx, const Tensor& input,
                                     IntArrayRef& dims, bool keep_dim) {
    // save tensor data to context
    ctx.data.emplace("input", input);
    ctx.data_int.emplace("keep_dim", keep_dim);
    Tensor out;
    DISPATCH_OP(sum_dim_impl, input.device(), input, out, dims, keep_dim);
    return {out};
  }

  static std::vector<Tensor> backward(Context& ctx,
                                      std::vector<Tensor>& grads) {
    Tensor& input = ctx.data.at("input");
    const Tensor& grad_output = grads[0];
    TORCH_CHECK(grad_output.numel() == 1,
                "grad_output numel should equal to 1");
    // y = a + b + c ..., y'(a) = 1 * grad[0], y'(b) = 1 * grad[1] ...
    Tensor grad_input = zeros(input.shape(), input.device());
    DISPATCH_OP(fill_impl, ctx.device, grad_input, grad_output[0]);
    return {grad_input};
  }
};

Tensor sum_dim(const Tensor& a, IntArrayRef dims, bool keep_dim) {
  return SumDimNode::forward_and_build_graph(a, dims, keep_dim)[0];
}

struct CloneNode : public FunctionNode<CloneNode> {
  static std::vector<Tensor> forward(Context& ctx, const Tensor& input) {
    Tensor out = zeros(input.shape(), input.device(), input.requires_grad());
    DISPATCH_OP(clone_impl, input.device(), input, out);
    return {out};
  }

  static std::vector<Tensor> backward(Context& ctx,
                                      std::vector<Tensor>& grads) {
    const Tensor& grad_output = grads[0];
    // y = a + b + c ..., y'(a) = 1 * grad[0], y'(b) = 1 * grad[1] ...
    Tensor grad_input = zeros(grad_output.shape(), grad_output.device());
    DISPATCH_OP(clone_backward_impl, ctx.device, grad_output, grad_input);
    return {grad_input};
  }
};

Tensor Tensor::clone() const {
  return CloneNode::forward_and_build_graph(*this)[0];
}

}  // namespace microtorch
