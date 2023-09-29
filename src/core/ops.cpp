
#include "ops.hpp"

#include "engine.hpp"
#include "graph.hpp"

namespace microtorch {

std::ostream& print_with_size(std::ostream& stream, Tensor t,
                              int64_t print_size,
                              const std::string& name = "name") {
  int64_t size = t.numel();
  TORCH_CHECK(t.device().is_cpu(), "only support print tensor in cpu now");
  TORCH_CHECK(t.shape().size() == 1, "only support 1d tensor print now");
  stream << "<microtorch.Tensor[" << name << "] size=" << size
         << ", device=" << t.device() << ", requires_grad=" << t.requires_grad()
         << ", next edge: " << (t.edge() ? t.edge()->node_name() : "no edge")
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

std::string print_with_size(Tensor t, int64_t print_size,
                            const std::string& name) {
  std::ostringstream s;
  print_with_size(s, t, print_size, name);
  return s.str();
}

std::ostream& operator<<(std::ostream& stream, Tensor t) {
  return print_with_size(stream, t, 20);
}

struct AddNode : public FunctionNode<AddNode> {
  static std::vector<Tensor> forward(Context& ctx, const Tensor& a,
                                     const Tensor& b) {
    Tensor out = zeros(a.shape(), a.device(), a.requires_grad());
    DISPATCH_OP(add_impl, a.device(), a, b, out);
    return {out};
  }
  static std::vector<Tensor> backward(Context& ctx,
                                      std::vector<Tensor>& grads) {
    auto& grad_output = grads[0];
    Tensor grad_input_1 = zeros(grad_output.shape(), grad_output.device());
    Tensor grad_input_2 = zeros(grad_output.shape(), grad_output.device());
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
Tensor Tensor::operator*(const data_t& other) {
  // TODO: to realize a mul kernel with scalar is better
  Tensor t = zeros(this->shape(), this->device());
  t.fill_(other);
  return MulNode::forward_and_build_graph(*this, t)[0];
}

Tensor& Tensor::operator*=(const Tensor& other) {
  check_device_shape({*this, other});
  TORCH_CHECK(
      !this->requires_grad() || !GradModeController::is_enabled(),
      "Tensor that requires grad is being used in an in-place operation");
  DISPATCH_OP(mul_impl, this->device(), *this, other, *this);
  return *this;
}
Tensor& Tensor::operator*=(const data_t& other) {
  Tensor t = zeros(this->shape(), this->device());
  t.fill_(other);
  TORCH_CHECK(
      !this->requires_grad() || !GradModeController::is_enabled(),
      "Tensor that requires grad is being used in an in-place operation");
  DISPATCH_OP(mul_impl, this->device(), *this, t, *this);
  return *this;
}

struct SquareNode : public FunctionNode<SquareNode> {
  static void forward(Context& ctx, std::vector<Tensor>& ins,
                      std::vector<Tensor>& outs) {
    ctx.data.emplace("input", ins[0]);
    DISPATCH_OP(square_impl, ctx.device, ins[0], outs[0]);
  }

  static std::vector<Tensor> backward(Context& ctx,
                                      std::vector<Tensor>& grads) {
    auto& input = ctx.data.at("input");
    auto& grad_output = grads[0];
    Tensor grad_input = zeros(input.shape(), input.device());
    DISPATCH_OP(square_backward_impl, ctx.device, grad_output, grad_input,
                input);
    return {grad_input};
  }
};

struct SumNode : public FunctionNode<SumNode> {
  static void forward(Context& ctx, std::vector<Tensor>& ins,
                      std::vector<Tensor>& outs) {
    ctx.data.emplace("input", ins[0]);
    DISPATCH_OP(sum_impl, ctx.device, ins[0], outs[0]);
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

struct CloneNode : public FunctionNode<CloneNode> {
  static void forward(Context& ctx, std::vector<Tensor>& ins,
                      std::vector<Tensor>& outs) {
    DISPATCH_OP(clone_impl, ctx.device, ins[0], outs[0]);
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

void initialize_ops() {
  OpRegistry::Instance().RegisterOp(
      "square", &(FunctionNode<SquareNode>::forward_and_build_graph));
  OpRegistry::Instance().RegisterOp(
      "sum", &(FunctionNode<SumNode>::forward_and_build_graph));
  OpRegistry::Instance().RegisterOp(
      "clone", &(FunctionNode<CloneNode>::forward_and_build_graph));
}
}  // namespace microtorch
