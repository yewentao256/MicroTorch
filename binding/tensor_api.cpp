#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../src/ops.hpp"
#include "../src/tensor.hpp"
namespace py = pybind11;
using namespace tinytorch;

void export_tensor_class(py::module &m) {
  py::class_<Tensor>(m, "Tensor")
      // init function
      .def(py::init<const Tensor &>())
      .def(py::init<std::vector<data_t>>(), py::arg("data"))
      .def("tensor_str",
           [](Tensor &t) { return tinytorch::print_with_size(t, 30, "name"); })
      .def(
          "__getitem__",
          [](Tensor &t, std::vector<size_t> idxs) { return t[idxs]; },
          py::is_operator())
      .def(
          "__setitem__",
          [](Tensor &t, std::vector<size_t> idxs, const data_t &value) {
            t[idxs] = value;
          },
          py::is_operator())
      .def(
          "__add__", [](Tensor &t1, Tensor &t2) { return t1 + t2; },
          py::is_operator())
      .def(
          "__mul__", [](Tensor &t1, Tensor &t2) { return t1 * t2; },
          py::is_operator())
      .def(
          "__sub__", [](Tensor &t1, Tensor &t2) { return t1 - t2; },
          py::is_operator())
      .def("__iadd__",
           [](Tensor &self, const Tensor &other) {
             self += other;
             return self;
           })
      .def("__isub__",
           [](Tensor &self, const Tensor &other) {
             self -= other;
             return self;
           })
      .def("__imul__",
           [](Tensor &self, const Tensor &other) {
             self *= other;
             return self;
           })

      // functions
      .def("is_contiguous", &Tensor::is_contiguous)
      .def("shape", &Tensor::shape)
      .def("grad", &Tensor::grad)
      .def("add_",
           [](Tensor &self, const Tensor &other) { return self += other; })
      .def("zero_", &Tensor::zero_)
      .def("fill_", &Tensor::fill_)
      .def("numel", &Tensor::numel)
      .def("device", [](Tensor &t) { return t.device().str(); })
      .def("cuda", &Tensor::cuda)
      .def("is_cuda", &Tensor::is_cuda)
      .def("cpu", &Tensor::cpu);
}

void export_tensor_function(py::module &m) {
  m.def("zeros", (Tensor(*)(size_t, const std::string &)) & tinytorch::zeros,
        "initialize a tensor with all zero", py::arg("size"), py::arg("device"))
      .def("zeros",
           (Tensor(*)(std::vector<size_t>, const std::string &)) &
               tinytorch::zeros,
           "initialize a tensor with random numbers", py::arg("shape"),
           py::arg("device"))
      .def("ones", (Tensor(*)(size_t, const std::string &)) & tinytorch::ones,
           "initialize a tensor with all one", py::arg("size"),
           py::arg("device"))
      .def("ones",
           (Tensor(*)(std::vector<size_t>, const std::string &)) &
               tinytorch::ones,
           "initialize a tensor with random numbers", py::arg("shape"),
           py::arg("device"))
      .def("rand", (Tensor(*)(size_t, const std::string &)) & tinytorch::rand,
           "initialize a tensor with random numbers", py::arg("size"),
           py::arg("device"))
      .def("rand",
           (Tensor(*)(std::vector<size_t>, const std::string &)) &
               tinytorch::rand,
           "initialize a tensor with random numbers", py::arg("shape"),
           py::arg("device"))
      .def("sum", &tinytorch::sum, "get the sum result of a tensor")
      .def("square", &tinytorch::square, "get the square result of a tensor");
}
