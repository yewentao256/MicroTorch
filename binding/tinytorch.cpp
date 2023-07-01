#include <pybind11/pybind11.h>
#include "../src/ops.hpp"

namespace py = pybind11;

void export_tensor_class(py::module &m);
void export_tensor_function(py::module &m);
void export_graph_function(py::module &m);
void export_optimizer_class(py::module &m);
void export_test_function(py::module &m);

PYBIND11_MODULE(tinytorch, m) {
    m.doc() = "TinyTorch: A simplest pytorch implementation for learning";
    
    export_tensor_class(m);
    export_tensor_function(m);
    export_graph_function(m);
    export_optimizer_class(m);
    export_test_function(m);

    tinytorch::initialize_ops();
}

