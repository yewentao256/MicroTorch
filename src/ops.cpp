
#include "ops.hpp"

#include "engine.hpp"
#include "graph.hpp"

namespace tinytorch {

// TODO: move this to another file
std::ostream& print_with_size(std::ostream& stream, Tensor t, size_t print_size,
                              const std::string& name = "name") {
  size_t size = t.numel();
  // TODO: support print tensor in cuda
  stream << "<tinytorch.Tensor[" << name << "] size=" << size
         << ", device=" << t.device() << ", storage_ptr: " << t.data_ptr()
         << ">: [";
  if (size > print_size) {
    // 只打印前print_size/2个和后print_size/2个元素
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
  return print_with_size(stream, t, 20);  // 默认打印20个元素
}

struct AddNode : public FunctionNode<AddNode> {
  static void forward(Context& ctx, std::vector<Tensor>& ins,
                      std::vector<Tensor>& outs) {
    DISPATCH_OP(add_impl, ctx.device, ins[0], ins[1], outs[0]);
  }
  static std::vector<Tensor> backward(Context& ctx, std::vector<Tensor>& ins) {
    auto dy = ins[0];
    Tensor dx_1 = zeros(dy.shape(), dy.device());
    Tensor dx_2 = zeros(dy.shape(), dy.device());
    DISPATCH_OP(add_backward_impl, ctx.device, dy, dx_1, dx_2);
    return {dx_1, dx_2};
  }
};

struct SubNode : public FunctionNode<SubNode> {
  static void forward(Context& ctx, std::vector<Tensor>& ins,
                      std::vector<Tensor>& outs) {
    // TODO: cuda
    sub_impl<Host>(ins[0], ins[1], outs[0]);
  }

  static std::vector<Tensor> backward(Context& ctx, std::vector<Tensor>& grad) {
    auto grad_a = sub_backward_impl(grad[0]);
    return grad_a;
  }
};

struct MulNode : public FunctionNode<MulNode> {
  static void forward(Context& ctx, std::vector<Tensor>& ins,
                      std::vector<Tensor>& outs) {
    // save tensor data to context
    ctx.data["t0"] = ins[0];
    ctx.data["t1"] = ins[1];
    mult_impl<Host>(ins[0], ins[1], outs[0]);
  }

  static std::vector<Tensor> backward(Context& ctx, std::vector<Tensor>& grad) {
    auto grad_a = mult_backward_impl(ctx.data["t0"], ctx.data["t1"], grad[0]);
    return grad_a;
  }
};
struct SquareNode : public FunctionNode<SquareNode> {
  static void forward(Context& ctx, std::vector<Tensor>& ins,
                      std::vector<Tensor>& outs) {
    ctx.data["t"] = ins[0];
    square_impl<Host>(ins[0], outs[0]);
  }

  static std::vector<Tensor> backward(Context& ctx, std::vector<Tensor>& grad) {
    auto grad_a = square_backward_impl(ctx.data["t"], grad[0]);
    return grad_a;
  }
};

struct SumNode : public FunctionNode<SumNode> {
  static void forward(Context& ctx, std::vector<Tensor>& ins,
                      std::vector<Tensor>& outs) {
    ctx.data["input"] = ins[0];
    sum_impl<Host>(ins[0], outs[0]);
  }

  static std::vector<Tensor> backward(Context& ctx,
                                      std::vector<Tensor>& grads) {
    TORCH_CHECK(grads.size() == 1, "grad size should equal to 1");
    auto grad_input = sum_backward_impl(ctx.data["input"], grads[0]);
    return grad_input;
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
