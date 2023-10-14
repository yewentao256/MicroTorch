#pragma once

#include <typeinfo>

#include "context.hpp"
#include "tensor.hpp"
#include "utils.hpp"

namespace microtorch {

class GradModeController {
 public:
  static void set_enabled(bool enabled) { grad_mode_enabled = enabled; }
  static bool is_enabled() { return grad_mode_enabled; }

 private:
  static inline bool grad_mode_enabled = true;
};

// RAII Guard
class AutoGradGuard {
 public:
  AutoGradGuard(bool enabled) : prev_mode(GradModeController::is_enabled()) {
    GradModeController::set_enabled(enabled);
  }
  ~AutoGradGuard() { GradModeController::set_enabled(prev_mode); }

 private:
  bool prev_mode;
};

struct Node;

struct Edge {
  // The function_node this `Edge` points to.
  std::shared_ptr<Node> function_node;

  // The input_nr of a particular input to the function_node.
  // For example, there are three inputs for function_node, then
  // input_nr can be 0, 1 or 2, representing the first, second
  // or third input of this function_node
  uint32_t input_nr;

  Edge(std::shared_ptr<Node> function_node, uint32_t input_nr) noexcept
      : function_node(std::move(function_node)), input_nr(input_nr) {}
};

struct Node {
  // A global counter to get correct node ordering
  int sequence_number;
  // Inline variable, see
  // https://stackoverflow.com/questions/38043442/how-do-inline-variables-work
  // Here we use it for unique increasing sequence number
  inline static int current_seq_nr;

  // The next edges are the inputs of the forward operator
  std::vector<std::shared_ptr<Edge>> next_edges;

  // Variables that are required for the backward pass
  Context context;

  // number of inputs for backward
  size_t num_input_of_backward;

  // Create a node and give it a unique increasing sequence number
  Node() : sequence_number(current_seq_nr++) {}

  // Computes and returns the gradients of the input tensor of the forward
  // operator. The input is the gradient of the forward output
  virtual std::vector<Tensor> backward(std::vector<Tensor>& forward_output) = 0;

  // for base class, it's recommended to use virtual deconstructor
  // so as to avoid memory leak
  virtual ~Node() = default;
};

template <typename T>
struct FunctionNode : public Node {
  FunctionNode() {}
  /* Deprecated and will be removed */
  static inline void infer_tensor(Context& ctx, std::vector<Tensor>& inputs) {
    int64_t len_inputs = inputs.size();
    Device device = inputs[0].device();
    auto shape = inputs[0].shape();
    for (int64_t i = 1; i < len_inputs; i++) {
      TORCH_CHECK(inputs[i].device() == device,
                  "all the tensors should be in the same device.");
      TORCH_CHECK(inputs[i].shape() == shape,
                  "size of the tensors should be the same");
    }
    ctx.device = device;
  }

  template <typename... Args>
  static std::vector<Tensor> forward_and_build_graph(Args&&... args) {
    // Check whether needs to build graph
    bool any_requires_grad = false;
    // Create node and set next edge
    auto node = std::make_shared<FunctionNode<T>>();
    auto inputs = make_arg_list(args...);

    if (GradModeController::is_enabled()) {
      for (const auto& arg : inputs) {
        if (std::holds_alternative<Tensor>(arg.value)) {
          auto& t = std::get<Tensor>(arg.value);
          if (t.requires_grad()) {
            node->next_edges.push_back(t.edge());
            any_requires_grad = true;
          }
        }
      }
    }

    // forward
    auto outs = T::forward(node->context, std::forward<Args>(args)...);

    if (any_requires_grad) {
      node->num_input_of_backward = outs.size();
      // Set the edges of the output to point to this node
      for (size_t i = 0; i < outs.size(); i++) {
        outs[i].set_requires_grad(true);
        outs[i].set_edge(std::make_shared<Edge>(node, i));
      }
    }
    return outs;
  }

  /* Deprecated and will be removed */
  static void forward_and_build_graph(std::vector<Tensor>& inputs,
                                      std::vector<Tensor>& outs) {
    // Check whether needs to build graph
    bool any_requires_grad = false;
    // Create node and set next edge
    auto node = std::make_shared<FunctionNode<T>>();
    infer_tensor(node->context, inputs);
    if (GradModeController::is_enabled()) {
      for (size_t i = 0; i < inputs.size(); i++) {
        // Here we bind the edge of tensor before to the current node
        if (inputs[i].requires_grad()) {
          node->next_edges.push_back(inputs[i].edge());
          any_requires_grad = true;
        }
      }
    }

    // forward
    T::forward(node->context, inputs, outs);

    if (any_requires_grad) {
      node->num_input_of_backward = outs.size();
      // Set the edges of the output to point to this node
      for (size_t i = 0; i < outs.size(); i++) {
        outs[i].set_requires_grad(true);
        outs[i].set_edge(std::make_shared<Edge>(node, i));
      }
    }
  }

  std::vector<Tensor> backward(std::vector<Tensor>& grad_outputs) override {
    TORCH_CHECK(grad_outputs.size() == num_input_of_backward,
                "grad_outputs num should equal to num_input_of_backward");
    return T::backward(context, grad_outputs);
  }
};

struct AccumulateGrad : public Node {
  // Each AccumulateGrad owns a tensor for calculating grad for updating params
  Tensor t;
  AccumulateGrad(Tensor t) : t(t) { num_input_of_backward = 1; }

  std::vector<Tensor> backward(std::vector<Tensor>& grad_outputs) override {
    TORCH_CHECK(grad_outputs.size() == 1,
                "grad_outputs size should equal to 1");
    if (t.grad().defined()) {
      t.grad() += grad_outputs[0];
    } else {
      t.set_grad(grad_outputs[0]);
    }
    return {};
  }
};

void backward(Tensor loss);

}  // namespace microtorch