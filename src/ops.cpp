
#include "ops.hpp"

#include "engine.hpp"
#include "graph.hpp"

namespace tinytorch {

std::ostream& print_with_size(std::ostream& stream, Tensor t, size_t print_size,
                              const std::string& name = "name") {
  size_t size = t.numel();
  TORCH_CHECK(t.device().is_cpu(), "only support print tensor in cpu now");
  TORCH_CHECK(t.shape().size() == 1, "only support 1d tensor print now");
  stream << "<tinytorch.Tensor[" << name << "] size=" << size
         << ", device=" << t.device() << ", requires_grad=" << t.requires_grad()
         << ", next edge: " << (t.edge() ? t.edge()->node_name() : "no edge")
         << ", storage_ptr: " << t.data_ptr() << ">: [";
  if (size > print_size) {
    for (size_t i = 0; i < print_size / 2; i++) {
      stream << std::setw(8) << t[i] << " ";
    }
    stream << " ... ";
    for (size_t i = size - print_size / 2; i < size; i++) {
      stream << std::setw(8) << t[i] << " ";
    }
  } else {
    for (size_t i = 0; i < size; i++) {
      stream << std::setw(8) << t[i] << " ";
    }
  }
  stream << "]";
  return stream;
}

std::string print_with_size(Tensor t, size_t print_size,
                            const std::string& name) {
  std::ostringstream s;
  print_with_size(s, t, print_size, name);
  return s.str();
}

std::ostream& operator<<(std::ostream& stream, Tensor t) {
  return print_with_size(stream, t, 20);
}

struct AddNode : public FunctionNode<AddNode> {
  static void forward(Context& ctx, std::vector<Tensor>& ins,
                      std::vector<Tensor>& outs) {
    DISPATCH_OP(add_impl, ctx.device, ins[0], ins[1], outs[0]);
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

struct SubNode : public FunctionNode<SubNode> {
  static void forward(Context& ctx, std::vector<Tensor>& ins,
                      std::vector<Tensor>& outs) {
    DISPATCH_OP(sub_impl, ctx.device, ins[0], ins[1], outs[0]);
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

struct MulNode : public FunctionNode<MulNode> {
  static void forward(Context& ctx, std::vector<Tensor>& ins,
                      std::vector<Tensor>& outs) {
    // save tensor data to context
    ctx.data.emplace("a", ins[0]);
    ctx.data.emplace("b", ins[1]);
    DISPATCH_OP(mul_impl, ctx.device, ins[0], ins[1], outs[0]);
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

void initialize_ops() {
  OpRegistry::Instance().RegisterOp(
      "add", &(FunctionNode<AddNode>::forward_and_build_graph));
  OpRegistry::Instance().RegisterOp(
      "sub", &(FunctionNode<SubNode>::forward_and_build_graph));
  OpRegistry::Instance().RegisterOp(
      "mul", &(FunctionNode<MulNode>::forward_and_build_graph));
  OpRegistry::Instance().RegisterOp(
      "square", &(FunctionNode<SquareNode>::forward_and_build_graph));
  OpRegistry::Instance().RegisterOp(
      "sum", &(FunctionNode<SumNode>::forward_and_build_graph));
}
}  // namespace tinytorch
