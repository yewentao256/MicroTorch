/**
 * Copyright (c) 2022-2024 yewentao256
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

// Reduce grad, matching the target shape
// This is needed in cases like broadcast, Tensor1([1, 2, 3]) + Tensor2([1]),
// The grad of Tensor2 should be [3] instead of [1, 1, 1]
Tensor reduce_grad_if_needed(Tensor t, const IntArrayRef& target_shape) {
  auto shape = t.shape();
  if (shape == target_shape) {
    return t;
  }
  if (target_shape.size() == 0) {
    return sum(t);
  }
  IntArrayRef reduce_dims;
  const int64_t leading_dims = shape.size() - target_shape.size();
  for (const auto i : irange(leading_dims)) {
    reduce_dims.push_back(i);
  }
  for (int64_t i = leading_dims; i < shape.size(); ++i) {
    if (target_shape[i - leading_dims] == 1 && shape[i] != 1) {
      reduce_dims.push_back(i);
    }
  }

  if (!reduce_dims.empty()) {
    t = sum_dim(t, reduce_dims, true);
  }
  if (leading_dims > 0) {
    t = t.as_strided(target_shape, calculate_init_stride(target_shape));
  }
  return t;
}

inline Tensor wrap_scalar_to_tensor(const data_t scalar, const Device& device) {
  return empty({}, device, false).fill_(scalar);
}

struct AddNode : public FunctionNode<AddNode> {
  static std::vector<Tensor> forward(Context& ctx, const Tensor& a,
                                     const Tensor& b) {
    // TODO: maybe we can save as a class instance variable
    ctx.data.emplace("a", a);
    ctx.data.emplace("b", b);
    TensorIterator iter;
    Tensor out;
    iter.add_output(out).add_input(a).add_input(b).build();
    DISPATCH_OP(add_impl, iter.common_device(), iter);
    return {iter.tensor(0)};
  }
  static std::vector<Tensor> backward(Context& ctx,
                                      std::vector<Tensor>& grads) {
    // y = a + b ..., y'(a) = y'(b) = 1 * grad
    TORCH_INTERNAL_ASSERT(grads.size() == 1);
    Tensor grad_input_1 =
        reduce_grad_if_needed(grads[0].clone(), ctx.data.at("a").shape());
    Tensor grad_input_2 =
        reduce_grad_if_needed(grads[0].clone(), ctx.data.at("b").shape());
    return {grad_input_1, grad_input_2};
  }
};

Tensor Tensor::operator+(const Tensor& other) const {
  return AddNode::forward_and_build_graph(*this, other)[0];
}

Tensor& Tensor::operator+=(const Tensor& other) {
  TORCH_CHECK(
      !this->requires_grad() || !GradModeController::is_enabled(),
      "Tensor that requires grad is being used in an in-place operation");
  TensorIterator iter;
  iter.add_output(*this).add_input(*this).add_input(other).build();
  DISPATCH_OP(add_impl, iter.common_device(), iter);
  return *this;
}

struct SubNode : public FunctionNode<SubNode> {
  static std::vector<Tensor> forward(Context& ctx, const Tensor& a,
                                     const Tensor& b) {
    ctx.data.emplace("a", a);
    ctx.data.emplace("b", b);
    TensorIterator iter;
    Tensor out;
    iter.add_output(out).add_input(a).add_input(b).build();
    DISPATCH_OP(sub_impl, iter.common_device(), iter);
    return {iter.tensor(0)};
  }
  static std::vector<Tensor> backward(Context& ctx,
                                      std::vector<Tensor>& grads) {
    TORCH_INTERNAL_ASSERT(grads.size() == 1);
    // y = a - b ..., y'(a) = 1 * grad, y'(b) = -1 * grad
    Tensor grad_input_1 =
        reduce_grad_if_needed(grads[0].clone(), ctx.data.at("a").shape());
    Tensor grad_input_2 =
        reduce_grad_if_needed(-grads[0], ctx.data.at("b").shape());
    return {grad_input_1, grad_input_2};
  }
};

Tensor Tensor::operator-(const Tensor& other) const {
  return SubNode::forward_and_build_graph(*this, other)[0];
}

Tensor& Tensor::operator-=(const Tensor& other) {
  TORCH_CHECK(
      !this->requires_grad() || !GradModeController::is_enabled(),
      "Tensor that requires grad is being used in an in-place operation");
  TensorIterator iter;
  iter.add_output(*this).add_input(*this).add_input(other).build();
  DISPATCH_OP(sub_impl, iter.common_device(), iter);
  return *this;
}

struct MulNode : public FunctionNode<MulNode> {
  static std::vector<Tensor> forward(Context& ctx, const Tensor& a,
                                     const Tensor& b) {
    ctx.data.emplace("a", a);
    ctx.data.emplace("b", b);
    TensorIterator iter;
    Tensor out;
    iter.add_output(out).add_input(a).add_input(b).build();
    DISPATCH_OP(mul_impl, iter.common_device(), iter);
    return {iter.tensor(0)};
  }

  static std::vector<Tensor> backward(Context& ctx,
                                      std::vector<Tensor>& grads) {
    TORCH_INTERNAL_ASSERT(grads.size() == 1);
    // y = a * b ..., y'(a) = b * grad, y'(b) = a * grad
    Tensor grad_input_1 = reduce_grad_if_needed(grads[0] * ctx.data.at("b"),
                                                ctx.data.at("a").shape());
    Tensor grad_input_2 = reduce_grad_if_needed(grads[0] * ctx.data.at("a"),
                                                ctx.data.at("b").shape());
    return {grad_input_1, grad_input_2};
  }
};

Tensor Tensor::operator*(const Tensor& other) const {
  return MulNode::forward_and_build_graph(*this, other)[0];
}
Tensor& Tensor::operator*=(const Tensor& other) {
  TORCH_CHECK(
      !this->requires_grad() || !GradModeController::is_enabled(),
      "Tensor that requires grad is being used in an in-place operation");
  TensorIterator iter;
  iter.add_output(*this).add_input(*this).add_input(other).build();
  DISPATCH_OP(mul_impl, iter.common_device(), iter);
  return *this;
}

struct MulScalarNode : public FunctionNode<MulScalarNode> {
  static std::vector<Tensor> forward(Context& ctx, const Tensor& a,
                                     const data_t b) {
    ctx.data.emplace("a", a);
    Tensor b_tensor = wrap_scalar_to_tensor(b, a.device());
    ctx.data.emplace("b", b_tensor);
    TensorIterator iter;
    Tensor out;
    iter.add_output(out).add_input(a).add_input(b_tensor).build();
    DISPATCH_OP(mul_impl, iter.common_device(), iter);
    return {iter.tensor(0)};
  }

  static std::vector<Tensor> backward(Context& ctx,
                                      std::vector<Tensor>& grads) {
    // y = a * b(scalar) ..., y'(a) = b * grad
    Tensor grad_input = reduce_grad_if_needed(
        grads[0] * ctx.data.at("b").item(), ctx.data.at("a").shape());
    return {grad_input};
  }
};

Tensor Tensor::operator*(const data_t other) const {
  return MulScalarNode::forward_and_build_graph(*this, other)[0];
}

Tensor& Tensor::operator*=(const data_t other) {
  TORCH_CHECK(
      !this->requires_grad() || !GradModeController::is_enabled(),
      "Tensor that requires grad is being used in an in-place operation");
  Tensor b_tensor = wrap_scalar_to_tensor(other, this->device());
  TensorIterator iter;
  Tensor out;
  iter.add_output(out).add_input(*this).add_input(b_tensor).build();
  DISPATCH_OP(mul_impl, iter.common_device(), iter);
  return *this;
}

struct DivNode : public FunctionNode<DivNode> {
  static std::vector<Tensor> forward(Context& ctx, const Tensor& a,
                                     const Tensor& b) {
    ctx.data.emplace("a", a);
    ctx.data.emplace("b", b);
    TensorIterator iter;
    Tensor out;
    iter.add_output(out).add_input(a).add_input(b).build();
    DISPATCH_OP(div_impl, iter.common_device(), iter);
    return {iter.tensor(0)};
  }

  static std::vector<Tensor> backward(Context& ctx,
                                      std::vector<Tensor>& grads) {
    // y = a / b, y'(a) = 1 / b * grad, y'(b) = -a * (1/b)^-2 * grad
    TORCH_INTERNAL_ASSERT(grads.size() == 1);
    auto& a = ctx.data.at("a");
    auto& b = ctx.data.at("b");
    Tensor grad_input_1 = reduce_grad_if_needed(grads[0] / b, a.shape());
    Tensor grad_input_2 =
        reduce_grad_if_needed(-grads[0] * a / b / b, b.shape());
    return {grad_input_1, grad_input_2};
  }
};

Tensor Tensor::operator/(const Tensor& other) const {
  return DivNode::forward_and_build_graph(*this, other)[0];
}
Tensor& Tensor::operator/=(const Tensor& other) {
  TORCH_CHECK(
      !this->requires_grad() || !GradModeController::is_enabled(),
      "Tensor that requires grad is being used in an in-place operation");
  TensorIterator iter;
  iter.add_output(*this).add_input(*this).add_input(other).build();
  DISPATCH_OP(div_impl, iter.common_device(), iter);
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
    TensorIterator iter;
    Tensor out;
    iter.add_output(out).add_input(input).build();
    DISPATCH_OP(clone_impl, iter.common_device(), iter);
    return {iter.tensor(0)};
  }

  static std::vector<Tensor> backward(Context& ctx,
                                      std::vector<Tensor>& grads) {
    // y = a, y'(a) = 1 * grad
    return grads;
  }
};

Tensor Tensor::clone() const {
  return CloneNode::forward_and_build_graph(*this)[0];
}

struct NegNode : public FunctionNode<NegNode> {
  static std::vector<Tensor> forward(Context& ctx, const Tensor& input) {
    TensorIterator iter;
    Tensor out;
    iter.add_output(out).add_input(input).build();
    DISPATCH_OP(neg_impl, iter.common_device(), iter);
    return {iter.tensor(0)};
  }

  static std::vector<Tensor> backward(Context& ctx,
                                      std::vector<Tensor>& grads) {
    // y = -x ..., y'(x) = -1 * grad
    return {grads[0] * -1};
  }
};

Tensor Tensor::operator-() const {
  return NegNode::forward_and_build_graph(*this)[0];
}

}  // namespace microtorch
