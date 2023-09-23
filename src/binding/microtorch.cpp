#include <pybind11/pybind11.h>
#include "ops.hpp"

namespace py = pybind11;

void export_tensor_class(py::module &m);
void export_tensor_function(py::module &m);
void export_optimizer_class(py::module &m);
void export_test_function(py::module &m);
void export_cuda_function(py::module &m);

PYBIND11_MODULE(_microtorch, m) {
    m.doc() = "MicroTorch: A simplest pytorch implementation for learning";
    
    export_tensor_class(m);
    export_tensor_function(m);
    export_optimizer_class(m);
    export_test_function(m);
    export_cuda_function(m);

    microtorch::initialize_ops();
}

