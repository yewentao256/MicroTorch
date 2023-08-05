#pragma once

#include "tensor.hpp"
#include "context.hpp"

namespace tinytorch {

struct Node;

struct Edge {
  // The function_node this `Edge` points to.
  std::shared_ptr<Node> function_node;

  // The input_identifier of a particular input to the function_node.
  // For example, there are three inputs for function_node, then
  // input_identifier can be 0, 1 or 2, representing the first, second
  // or third input of this function_node
  uint32_t input_identifier;

  Edge(std::shared_ptr<Node> function_node, uint32_t input_identifier) noexcept
      : function_node(std::move(function_node)), input_identifier(input_identifier) {}
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
  static inline void infer_tensor(Context& ctx, std::vector<Tensor>& inputs) {
    size_t len_inputs = inputs.size();
    Device device = inputs[0].device();
    auto shape = inputs[0].shape();
    for(size_t i = 1; i < len_inputs; i++){
      TORCH_CHECK(inputs[i].device() == device, "all the tensors should be in the same device.");
      TORCH_CHECK(inputs[i].shape() == shape, "size of the tensors should be the same");
      // TODO: support broadcast
    }
    ctx.device = device;
  }

  static void forward_and_build_graph(
      std::vector<Tensor>& inputs, std::vector<Tensor>& outs) {
    // Create node and set next edge
    auto node = std::make_shared<FunctionNode<T>>();
    infer_tensor(node->context, inputs);
    for (size_t i = 0; i < inputs.size(); i++) {
      // Here we bind the edge of tensor before to the current node
      node->next_edges.push_back(inputs[i].get_edge());
    }

    // forward
    T::forward(node->context, inputs, outs);

    node->num_input_of_backward = outs.size();

    // Set the edges of the output to point to this node
    for (size_t i = 0; i < outs.size(); i++) {
      outs[i].set_edge(std::make_shared<Edge>(node, i));
    }
  }

  std::vector<Tensor> backward(std::vector<Tensor>& grad_outputs) override {
    TORCH_CHECK(grad_outputs.size() == num_input_of_backward, "grad_outputs num should equal to num_input_of_backward");
    return T::backward(context, grad_outputs);
  }
};

struct AccumulateGrad : public Node {
  // Each AccumulateGrad owns a tensor for calculating grad for updating params
  Tensor t;
  AccumulateGrad(Tensor t) : t(t) { num_input_of_backward = 1; }

  std::vector<Tensor> backward(std::vector<Tensor>& grad_outputs) override {
    TORCH_CHECK(grad_outputs.size() == 1, "grad_outputs size should equal to 1");
    t.grad() += grad_outputs[0];
    return {};
  }
};

inline void makeParameter(Tensor t) {
  t.set_edge(std::make_shared<Edge>(std::make_shared<AccumulateGrad>(t), 0));
}

}  // namespace tinytorch