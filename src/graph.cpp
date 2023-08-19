#include "graph.hpp"

namespace tinytorch {

const char* Edge::node_name() {
  if (function_node) {
    return typeid(*function_node).name();
  } else {
    return "no function node";
  }
}

}  // namespace tinytorch