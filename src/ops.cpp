
#include "ops.hpp"

#include "cuda_impl.hpp"
#include "graph.hpp"
#include "log.hpp"

#ifndef USE_CUDA
#define DISPATCH_OP_AUTO(host_func, device_func, ...)              \
  log_once([]() {},                                                \
           []() {                                                  \
             std::cout << "LOG: using host func `" #host_func "` " \
                       << std::endl;                               \
           });                                                     \
  auto result = host_func(__VA_ARGS__);
#else
#define DISPATCH_OP_AUTO(host_func, device_func, ...)                   \
  log_once([]() {},                                                     \
           []() {                                                       \
             std::cout << "LOG: using device func ` " #device_func "` " \
                       << std::endl;                                    \
           });                                                          \
  auto result = device_func(__VA_ARGS__);
#endif

namespace tinytorch {
struct AddNode : public FunctionNode<AddNode> {
  static std::vector<Tensor> forward(Context &ctx, std::vector<Tensor> t_lst) {
    // DISPATCH_OP_AUTO(add_impl, add_cuda_impl, t_lst[0], t_lst[1]);
    Tensor result;
    if (t_lst[0].arch() != "cuda"){
      result = add_impl(t_lst[0], t_lst[1]);
    }
    else {
      result = add_cuda_impl(t_lst[0], t_lst[1]);
      result.cuda();
    }
    return {result};
  }
  static std::vector<Tensor> backward(Context &ctx, std::vector<Tensor> grad) {
    auto grad_a = add_backward_impl(grad[0]);
    // DISPATCH_OP_AUTO(add_backward_impl, add_backward_cuda_impl, grad[0]);

    return grad_a;
  }
};

Tensor operator+(Tensor a, Tensor b) {
  // forward returns {result}, only one element.
  return FunctionNode<AddNode>::forward_and_build_graph({a, b})[0];
}

struct SubNode : public FunctionNode<SubNode> {
  static std::vector<Tensor> forward(Context &ctx, std::vector<Tensor> t_lst) {
    auto result = sub_impl(t_lst[0], t_lst[1]);
    return {result};
  }

  static std::vector<Tensor> backward(Context &ctx, std::vector<Tensor> grad) {
    auto grad_a = sub_backward_impl(grad[0]);
    return grad_a;
  }
};
Tensor operator-(Tensor a, Tensor b) {
  return FunctionNode<SubNode>::forward_and_build_graph({a, b})[0];
}

struct MultNode : public FunctionNode<MultNode> {
  static std::vector<Tensor> forward(Context &ctx, std::vector<Tensor> t_lst) {
    // save tensor data to context
    ctx.data["t0"] = t_lst[0];
    ctx.data["t1"] = t_lst[1];
    auto result = mult_impl(t_lst[0], t_lst[1]);
    return {result};
  }

  static std::vector<Tensor> backward(Context &ctx, std::vector<Tensor> grad) {
    auto grad_a = mult_backward_impl(ctx.data["t0"], ctx.data["t1"], grad[0]);
    return grad_a;
  }
};
Tensor operator*(Tensor a, Tensor b) {
  return FunctionNode<MultNode>::forward_and_build_graph({a, b})[0];
}

struct SquareNode : public FunctionNode<SquareNode> {
  static std::vector<Tensor> forward(Context &ctx, std::vector<Tensor> t_lst) {
    ctx.data["t"] = t_lst[0];
    auto result = square_impl(t_lst[0]);
    return {result};
  }

  static std::vector<Tensor> backward(Context &ctx, std::vector<Tensor> grad) {
    auto grad_a = square_backward_impl(ctx.data["t"], grad[0]);
    return grad_a;
  }
};
Tensor square(Tensor a) {
  return FunctionNode<SquareNode>::forward_and_build_graph({a})[0];
}

struct SumNode : public FunctionNode<SumNode> {
  static std::vector<Tensor> forward(Context &ctx, std::vector<Tensor> t_lst) {
    ctx.data_int["size"] = t_lst[0].size();
    auto result = sum_impl(t_lst[0]);
    return {result};
  }

  static std::vector<Tensor> backward(Context &ctx, std::vector<Tensor> grad) {
    assert(grad.size() == 1);
    auto grad_a = sum_backward_impl(ctx.data_int["size"], grad[0]);
    return grad_a;
  }
};

Tensor sum(Tensor a) {
  return FunctionNode<SumNode>::forward_and_build_graph({a})[0];
}
}  // namespace tinytorch
