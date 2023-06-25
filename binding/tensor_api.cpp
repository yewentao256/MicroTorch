#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../src/ops.hpp"
#include "../src/tensor.hpp"
namespace py = pybind11;
using namespace tinytorch;

void export_tensor_class(py::module &m) {
  py::class_<Tensor>(m, "Tensor")
      // init function
      .def(py::init<int>(), py::arg("size"))
      // .def(py::init<std::vector<float>>(), py::arg("data"))

      // python specs
      .def("__repr__",
           [](Tensor &t) {
             // TODO:
             // 在python层做封装可以实现任意size的打印。或者也可以考虑新建一个print_tensor方法，因为__repr__只支持一个self参数
             return tinytorch::repr(t, 30, "name");
           })
      .def(
          "__getitem__", [](Tensor &t, int i) { return t[i]; },
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

      // functions
      .def("transpose", &Tensor::transpose, py::arg("dim0"), py::arg("dim1"))
      .def("permute", &Tensor::permute, py::arg("dims"))
      .def("is_contiguous", &Tensor::is_contiguous)
      .def("size", &Tensor::size)
      .def("resize", &Tensor::resize, py::arg("size"))
      .def("clear_grad", &Tensor::clearGrad)
      .def("grad", &Tensor::grad)
      .def("add_", &Tensor::addInplace)
      .def("add_grad_", &Tensor::addGradInplace)
      .def("cuda", &Tensor::cuda)
      .def("cpu", &Tensor::cpu);
}

void export_tensor_function(py::module &m) {
  m.def("zeros", &tinytorch::zeros, "initialize a tensor with all zero",
        py::arg("size"), py::arg("device") = "cpu")
      .def("ones", &tinytorch::ones, "initialize a tensor with all one",
           py::arg("size"), py::arg("device") = "cpu")
      .def("rand", &tinytorch::rand, "initialize a tensor with random numbers",
           py::arg("size"), py::arg("device") = "cpu")
      .def("sum", &tinytorch::sum, "get the sum result of a tensor")
      .def("square", &tinytorch::square, "get the square result of a tensor");
}
