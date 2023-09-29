#include "binding.hpp"
#include "ops.hpp"

PYBIND11_MODULE(_microtorch, m) {
  m.doc() = "MicroTorch: A simplest pytorch implementation for learning";

  export_tensor_class(m);
  export_tensor_function(m);
  export_optimizer_class(m);
  export_test_function(m);
  export_cuda_function(m);
  export_graph_function(m);

  microtorch::initialize_ops();
}
