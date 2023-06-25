#pragma once
#include "graph.hpp"
#include "ops.hpp"
#include "tensor.hpp"

namespace tinytorch {

void backward(Tensor loss) {
  // for graph traversal
  std::vector<std::shared_ptr<Node>> node_stack;

  // temple variable for accumulating the gradients
  std::map<std::shared_ptr<Node>, std::vector<Tensor>> grad_map;

  // TODO: this is size or numel?
  TORCH_CHECK(loss.size() == 1, "loss size should equal to 1");
  TORCH_CHECK(loss.getEdge(), "loss should have edge");

  // start traversal at the root node
  std::shared_ptr<Node> root_node = (*loss.getEdge()).function;
  node_stack.push_back(root_node);

  // Normally the gradient of the final loss is 1
  Tensor one = ones(1);
  grad_map[root_node] = {one};

  while (!node_stack.empty()) {
    // sort by sequence number
    std::sort(node_stack.begin(), node_stack.end(), [](auto &n1, auto &n2) {
      return (*n1).sequence_number < (*n2).sequence_number;
    });

    std::shared_ptr<Node> current_node = node_stack.back();  // last node
    node_stack.pop_back();

    // backpropagate gradients
    std::vector<Tensor> next_gradients =
        (*current_node).backward(grad_map[current_node]);

    // Traverse to next nodes
    for (size_t i = 0; i < (*current_node).next.size(); i++) {
      auto next = (*current_node).next[i];
      if (next) {
        auto next_node = next->function;
        auto& next_tensor = next_gradients[next->identifier];

        // accumulate the gradient
        grad_map[next_node].resize(next_node->num_input_of_backward);
        if (grad_map[next_node][next->identifier].size() == 0){
          // initialization
          grad_map[next_node][next->identifier] = zeros(next_tensor.size(), next_tensor.arch());
        }
        grad_map[next_node][next->identifier].addInplace(next_tensor);

        // add next node to the stack
        node_stack.push_back(next->function);
      }
    }
  }
}

}  // namespace tinytorch