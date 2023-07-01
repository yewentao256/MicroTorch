
#include "ops.hpp"

#include "engine.hpp"
#include "graph.hpp"

#ifdef USE_CUDA
#define DISPATCH_OP_AUTO(func, ctx, ...) \
  if (ctx.device.is_cpu()) {             \
    func<Host>(ctx, __VA_ARGS__);        \
  } else {                               \
    func<Cuda>(ctx, __VA_ARGS__);        \
  }
#else
#define DISPATCH_OP_AUTO(func, ctx, ...)                                 \
  if (ctx.device.is_cpu()) {                                             \
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
    Tensor dx_1 = zeros(dy.size(), dy.device());
    Tensor dx_2 = zeros(dy.size(), dy.device());
    DISPATCH_OP_AUTO(add_backward_impl, ctx, dy, dx_1, dx_2);
    return {dx_1, dx_2};
  }
};

inline void add_out(const Tensor& a, const Tensor& b, Tensor& out) {
  OpNode("add").ins({a, b}).outs({out}).apply();
}

Tensor Tensor::operator+(const Tensor& other) {
  Tensor out = zeros(this->size(), this->device());
  add_out(*this, other, out);
  return out;
}

Tensor& Tensor::operator+=(const Tensor& other) {
  add_out(*this, other, *this);
  return *this;
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

inline void sub_out(const Tensor& a, const Tensor& b, Tensor& out) {
  OpNode("sub").ins({a, b}).outs({out}).apply();
}

Tensor Tensor::operator-(const Tensor& other) {
  // TODO：注意这里operator-返回一个新的Tensor对象，而不是引用。这可以通过返回值优化（Return
  // Value Optimization，RVO）或命名返回值优化（Named Return Value
  // Optimization，NRVO）来高效地完成
  // 对于Tensor这样的类型，通常会禁用复制构造函数，因为复制一个Tensor可能会涉及到大量数据的复制，这可能会非常耗时。相反，Tensor通常会提供移动构造函数。这也是可以被RVO来优化的
  // 需要检查一下目前tensor的复制构造和移动构造
  Tensor out = zeros(this->size(), this->device());
  sub_out(*this, other, out);
  return out;
}

Tensor& Tensor::operator-=(const Tensor& other) {
  sub_out(*this, other, *this);
  return *this;
}

struct MulNode : public FunctionNode<MulNode> {
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
  Tensor out = zeros(a.size(), a.device());
  OpNode("mul").ins({a, b}).outs({out}).apply();
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
  Tensor out = zeros(a.size(), a.device());
  OpNode("square").ins({a}).outs({out}).apply();
  return out;
}

struct SumNode : public FunctionNode<SumNode> {
  static void forward(Context& ctx, std::vector<Tensor>& ins,
                      std::vector<Tensor>& outs) {
    ctx.data_int["size"] = ins[0].size();
    sum_impl<Host>(ctx, ins[0], outs[0]);
  }

  static std::vector<Tensor> backward(Context& ctx, std::vector<Tensor>& grad) {
    TORCH_CHECK(grad.size() == 1, "grad size should equal to 1");
    auto grad_a = sum_backward_impl(ctx.data_int["size"], grad[0]);
    return grad_a;
  }
};

Tensor sum(Tensor& a) {
  Tensor out = zeros(1);
  OpNode("sum").ins({a}).outs({out}).apply();
  return out;
}

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
