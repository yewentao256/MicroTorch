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

  assert(loss.size() == 1);
  assert(loss.getEdge());

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

        // accumulate the gradient
        grad_map[next_node].resize(next_node->num_input_of_backward);
        grad_map[next_node][next->identifier].addInplace(
            next_gradients[next->identifier]);

        // add next node to the stack
        node_stack.push_back(next->function);
      }
    }
  }
}

}  // namespace tinytorch