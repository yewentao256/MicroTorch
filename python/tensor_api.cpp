#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../src/ops.hpp"
#include "../src/tensor.hpp"
namespace py = pybind11;
using namespace tinytorch;

void export_tensor_class(py::module &m)
{
    py::class_<Tensor>(m, "Tensor")
        // init function
        .def(py::init<int>(), py::arg("size"))
        .def(py::init<std::vector<float>>(), py::arg("data"))

        // python specs
        .def("__repr__", [](Tensor &t)
             { return tinytorch::repr(t); })
        .def(
            "__getitem__", [](Tensor &t, int i)
            { return t[i]; },
            py::is_operator())
        .def(
            "__add__", [](Tensor &t1, Tensor &t2)
            { return t1 + t2; },
            py::is_operator())
        .def(
            "__mul__", [](Tensor &t1, Tensor &t2)
            { return t1 * t2; },
            py::is_operator())
        .def(
            "__sub__", [](Tensor &t1, Tensor &t2)
            { return t1 - t2; },
            py::is_operator())

        // functions
        .def("size", &Tensor::size)
        .def("resize", &Tensor::resize, py::arg("size"))
        .def("clear_grad", &Tensor::clearGrad)
        .def("grad", &Tensor::grad)
        .def("add_", &Tensor::addInplace)
        .def("add_grad_", &Tensor::addGradInplace);
}

void export_tensor_function(py::module &m)
{
    m.def("zero", &tinytorch::zero, "initialize a tensor with all zero",
          py::arg("size"))
    .def("ones", &tinytorch::ones, "initialize a tensor with all ones",
            py::arg("size"))
    .def("rand", &tinytorch::rand, "initialize a tensor with random numbers",
            py::arg("size"))
    .def("sum", &tinytorch::sum, "get the sum result of a tensor")
    .def("square", &tinytorch::square, "get the square result of a tensor");
}
