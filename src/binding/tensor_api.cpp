/**
 * Copyright (c) 2022-2023 yewentao
 * Licensed under the MIT License.
 */
#include "binding.hpp"
#include "ops.hpp"

using namespace microtorch;

void export_tensor_class(py::module &m) {
  py::class_<Tensor>(m, "Tensor")
      // init function
      .def(py::init<const Tensor &>())
      .def(py::init<std::vector<data_t>, const std::string, bool>(),
           py::arg("data"), py::arg("device") = "cpu",
           py::arg("requires_grad") = false)
      .def("str", [](Tensor &t) { return t.str(); })

      // magic methods
      .def(
          "__getitem__",
          [](const Tensor &t, IntArrayRef idxs) { return t[idxs]; },
          py::is_operator())
      .def(
          "__setitem__",
          [](Tensor &t, IntArrayRef idxs, const data_t &value) {
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
          "__mul__", [](Tensor &t1, const float other) { return t1 * other; },
          py::is_operator())
      .def(
          "__truediv__", [](Tensor &t1, Tensor &t2) { return t1 / t2; },
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
      .def("__imul__",
           [](Tensor &self, const float other) {
             self *= other;
             return self;
           })
      .def("__itruediv__",
           [](Tensor &self, const Tensor &other) {
             self /= other;
             return self;
           })
      .def("__eq__",
           [](Tensor &self, const Tensor &other) { return self == other; })

      // properties
      .def_property(
          "requires_grad",
          [](const Tensor &t) { return t.requires_grad(); },  // Getter
          [](Tensor &t, bool requires_grad) {
            t.set_requires_grad(requires_grad);
          }  // Setter
          )

      // functions
      .def("is_contiguous", &Tensor::is_contiguous)
      .def("shape", &Tensor::shape)
      .def("grad", &Tensor::grad)
      .def("backward", &Tensor::backward)
      .def("zero_grad", [](Tensor &t) { return t.grad().zero_(); })
      .def("zero_", &Tensor::zero_)
      .def("fill_", &Tensor::fill_)
      .def("clone", &Tensor::clone)
      .def("numel", &Tensor::numel)
      .def("device", [](Tensor &t) { return t.device().str(); })
      .def("cuda", &Tensor::cuda)
      .def("is_cuda", &Tensor::is_cuda)
      .def("cpu", &Tensor::cpu)
      .def("square", &Tensor::square)
      .def("equal", &Tensor::equal)
      .def("defined", &Tensor::defined);
}

void export_tensor_function(py::module &m) {
  m.def("empty", &microtorch::empty)
      .def("zeros", &microtorch::zeros)
      .def("ones", &microtorch::ones)
      .def("rand", &microtorch::rand)
      .def("sum", &microtorch::sum, "get the sum result of a tensor")
      .def("square", &microtorch::square, "get the square result of a tensor");
}
