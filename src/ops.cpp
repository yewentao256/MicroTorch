
#include "ops.hpp"

#include "engine.hpp"
#include "graph.hpp"

#ifdef USE_CUDA
#define DISPATCH_OP_AUTO(func, ctx, ...) \
  if (ctx.arch == "cpu") {              \
    func<Host>(ctx, __VA_ARGS__);        \
  } else {                               \
    func<Cuda>(ctx, __VA_ARGS__);        \
  }
#else
#define DISPATCH_OP_AUTO(func, ctx, ...)                                 \
  if (ctx.arch == "cpu") {                                              \
    func<Host>(ctx, __VA_ARGS__);                                        \
  } else {                                                               \
    std::cout << "Not support device in host compile mode" << std::endl; \
  }                                                                      \
  // auto result = device_func(__VA_ARGS__);
#endif

namespace tinytorch {
struct AddNode : public FunctionNode<AddNode> {
  static void forward(Context& ctx, std::vector<Tensor>& ins,
                      std::vector<Tensor>& outs) {
    DISPATCH_OP_AUTO(add_impl, ctx, ins[0], ins[1], outs[0]);
  }
  static std::vector<Tensor> backward(Context& ctx, std::vector<Tensor>& ins) {
    auto dy = ins[0];
    Tensor dx_1 = zeros(dy.size(), dy.arch());
    Tensor dx_2 = zeros(dy.size(), dy.arch());
    // add_backward_impl<Host>(dy, dx_1, dx_2);
    DISPATCH_OP_AUTO(add_backward_impl, ctx, dy, dx_1, dx_2);
    return {dx_1, dx_2};
  }
};

Tensor operator+(Tensor& a, Tensor& b) {
  Tensor out = zeros(a.size(), a.arch());
  ApplyFunc func = &(FunctionNode<AddNode>::forward_and_build_graph);
  OpNode("add").ins({a, b}).outs({out}).apply(func);
  return out;
}

struct SubNode : public FunctionNode<SubNode> {
  static void forward(Context& ctx, std::vector<Tensor>& ins,
                      std::vector<Tensor>& outs) {
    // TODO: cuda
    sub_impl<Host>(ctx, ins[0], ins[1], outs[0]);
  }

  static std::vector<Tensor> backward(Context& ctx, std::vector<Tensor>& grad) {
    auto grad_a = sub_backward_impl(grad[0]);
    return grad_a;
  }
};
Tensor operator-(Tensor& a, Tensor& b) {
  Tensor out = zeros(a.size(), a.arch());
  ApplyFunc func = &(FunctionNode<SubNode>::forward_and_build_graph);
  OpNode("sub").ins({a, b}).outs({out}).apply(func);
  return out;
}

struct MultNode : public FunctionNode<MultNode> {
  static void forward(Context& ctx, std::vector<Tensor>& ins,
                      std::vector<Tensor>& outs) {
    // save tensor data to context
    ctx.data["t0"] = ins[0];
    ctx.data["t1"] = ins[1];
    mult_impl<Host>(ctx, ins[0], ins[1], outs[0]);
  }

  static std::vector<Tensor> backward(Context& ctx, std::vector<Tensor>& grad) {
    auto grad_a = mult_backward_impl(ctx.data["t0"], ctx.data["t1"], grad[0]);
    return grad_a;
  }
};
Tensor operator*(Tensor& a, Tensor& b) {
  Tensor out = zeros(a.size(), a.arch());
  ApplyFunc func = &(FunctionNode<MultNode>::forward_and_build_graph);
  OpNode("add").ins({a, b}).outs({out}).apply(func);
  return out;
}

struct SquareNode : public FunctionNode<SquareNode> {
  static void forward(Context& ctx, std::vector<Tensor>& ins,
                      std::vector<Tensor>& outs) {
    ctx.data["t"] = ins[0];
    square_impl<Host>(ctx, ins[0], outs[0]);
  }

  static std::vector<Tensor> backward(Context& ctx, std::vector<Tensor>& grad) {
    auto grad_a = square_backward_impl(ctx.data["t"], grad[0]);
    return grad_a;
  }
};
Tensor square(Tensor& a) {
  Tensor out = zeros(a.size(), a.arch());
  ApplyFunc func = &(FunctionNode<SquareNode>::forward_and_build_graph);
  OpNode("square").ins({a}).outs({out}).apply(func);
  return out;
}

struct SumNode : public FunctionNode<SumNode> {
  static void forward(Context& ctx, std::vector<Tensor>& ins,
                      std::vector<Tensor>& outs) {
    ctx.data_int["size"] = ins[0].size();
    sum_impl<Host>(ctx, ins[0], outs[0]);
  }

  static std::vector<Tensor> backward(Context& ctx, std::vector<Tensor>& grad) {
    assert(grad.size() == 1);
    auto grad_a = sum_backward_impl(ctx.data_int["size"], grad[0]);
    return grad_a;
  }
};

Tensor sum(Tensor& a) {
  Tensor out = zeros(1);
  ApplyFunc func = &(FunctionNode<SumNode>::forward_and_build_graph);
  OpNode("sum").ins({a}).outs({out}).apply(func);
  return out;
}
}  // namespace tinytorch
