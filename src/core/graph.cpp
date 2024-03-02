/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#include "graph.hpp"

#include <unordered_set>

#include "ops.hpp"

namespace microtorch {

void backward(Tensor loss) {
  if (!GradModeController::is_enabled()) {
    return;
  }
  // for graph traversal
  std::vector<std::shared_ptr<Node>> node_stack;

  // temp variable for accumulating the gradients
  std::map<std::shared_ptr<Node>, std::vector<Tensor>> grad_map;

  // Set to store nodes that have been added to the stack
  std::unordered_set<std::shared_ptr<Node>> visited_nodes;

  TORCH_CHECK(loss.numel() == 1, "loss size should equal to 1");
  TORCH_CHECK(loss.edge(), "loss should have edge");

  // start traversal at the root node
  std::shared_ptr<Node> root_node = loss.edge()->function_node;
  node_stack.push_back(root_node);
  visited_nodes.insert(root_node);  // Mark root node as visited

  // Start with the gradient of 1
  Tensor one = ones({1}, loss.device(), false);
  grad_map[root_node] = {one};

  while (!node_stack.empty()) {
    // sort by sequence number
    std::sort(node_stack.begin(), node_stack.end(), [](auto &n1, auto &n2) {
      return n1->sequence_number < n2->sequence_number;
    });

    // last node (the highest sequence number)
    std::shared_ptr<Node> current_node = node_stack.back();
    node_stack.pop_back();

    // backpropagate, these gradients will be added to the next node
    std::vector<Tensor> next_gradients =
        current_node->backward(grad_map[current_node]);

    // Traverse to next nodes
    for (size_t i = 0; i < current_node->next_edges.size(); i++) {
      auto next_edge = current_node->next_edges[i];
      if (next_edge) {
        auto next_node = next_edge->function_node;
        auto &next_gradient = next_gradients[i];

        // resize vector<Tensor> to store the gradients of next node
        grad_map[next_node].resize(next_node->num_input_of_backward);
        if (!(grad_map[next_node][next_edge->input_nr].defined())) {
          // if tensor is not defined, initialization with zeros
          grad_map[next_node][next_edge->input_nr] =
              zeros(next_gradient.shape(), next_gradient.device());
        }
        // Accumulate the gradient according to edge's identifier.
        // For example, the next node takes three inputs(namely `grad_outputs`)
        // for backward, and there are three edges pointing to the node,
        // if the edge's identifier == 1, it means that this is the second edge,
        // and this next_gradient is the second input of the next node.
        grad_map[next_node][next_edge->input_nr] += next_gradient;

        // add next node to the stack only if it's not been added before
        if (visited_nodes.find(next_edge->function_node) ==
            visited_nodes.end()) {
          node_stack.push_back(next_edge->function_node);
          visited_nodes.insert(
              next_edge->function_node);  // Mark node as visited
        }
      }
    }
  }
}

}  // namespace microtorch