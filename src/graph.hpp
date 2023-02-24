#pragma once

#include "tensor.hpp"
#include "context.hpp"

namespace tinytorch {

struct Node;

struct Edge {
  /// The function this `Edge` points to.
  std::shared_ptr<Node> function;
  /// The identifier of a particular input to the function.
  uint32_t identifier;

  Edge(std::shared_ptr<Node> function, uint32_t identifier) noexcept
      : function(std::move(function)), identifier(identifier) {}
};

struct Node {
  // A global counter to get correct node ordering
  int sequence_number;
  // Inline variable, see
  // https://stackoverflow.com/questions/38043442/how-do-inline-variables-work
  // Here we use it for unique increasing sequence number
  inline static int current_seq_nr;

  // The next edges are the inputs of the forward operator
  std::vector<std::shared_ptr<Edge>> next;

  // Variables that are required for the backward pass
  Context context;

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
  static inline std::string infer_tensor(std::vector<Tensor> ins) {
    size_t len_ins = ins.size();
    std::string arch = ins[0].arch();
    size_t size = ins[0].size();
    for(size_t i = 1; i < len_ins; i++){
      assert(ins[i].arch() == arch);
      assert(ins[i].size() == size);
      // TODO: support broadcast
    }
    return arch;
  }

  static void forward_and_build_graph(
      std::vector<Tensor>& ins, std::vector<Tensor>& outs) {
    // Create node and set next edge
    auto node = std::make_shared<FunctionNode<T>>();
    node->context.arch = infer_tensor(ins);
    for (size_t i = 0; i < ins.size(); i++) {
      // Here we bind the edge of tensor before to the current node
      node->next.push_back(ins[i].getEdge());
    }

    // forward
    T::forward(node->context, ins, outs);

    node->num_input_of_backward = outs.size();

    // Set the edges of the output to point to this node
    for (size_t i = 0; i < outs.size(); i++) {
      outs[i].setEdge(std::make_shared<Edge>(node, i));
    }
  }

  std::vector<Tensor> backward(std::vector<Tensor>& forward_outputs) override {
    assert(forward_outputs.size() == num_input_of_backward);
    return T::backward(context, forward_outputs);
  }
};

struct AccumulateGrad : public Node {
  // Each AccumulateGrad owns a tensor for calculating grad
  // Usually for updating params
  Tensor t;
  AccumulateGrad(Tensor t) : t(t) { num_input_of_backward = 1; }

  std::vector<Tensor> backward(std::vector<Tensor>& input_grad) override {
    assert(input_grad.size() == 1);
    t.addGradInplace(input_grad[0]);
    return {};
  }
};

inline void makeParameter(Tensor t) {
  t.setEdge(std::make_shared<Edge>(std::make_shared<AccumulateGrad>(t), 0));
}

}  // namespace tinytorch