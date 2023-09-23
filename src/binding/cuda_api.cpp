#include <pybind11/pybind11.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#define CUDA_AVAILABLE true
#else
#define CUDA_AVAILABLE false
#endif

namespace py = pybind11;

void cuda_synchronize() {
#ifdef USE_CUDA
  cudaDeviceSynchronize();
#endif
}

void export_cuda_function(py::module &m) {
  m.def(
       "is_cuda_available", []() { return CUDA_AVAILABLE; },
       "Check if CUDA is available")
      .def("cuda_synchronize", &cuda_synchronize, "Synchronize CUDA device");
}
