# MicroTorch

English | [简体中文](README.zh-cn.md)

**MicroTorch**: Demystifying the Magic of Pytorch

Have you ever been curious about the underlying principles of PyTorch? Or have you ever wanted to build an operator from scratch?

Welcome to MicroTorch!

Gone are the intimidating complexities. MicroTorch offers a simplified, intuitive implementation to get you started with the essence of deep learning. You will understand how computational graphs are constructed, how automatic differentiation is implemented, and possess the capabilities for basic CUDA operator development.

## Features

With MicroTorch, you can:

- Build your own **Tensor** object
- Understand the forward and backward computation processes of basic operators
- Grasp the construction of computational graphs and the process of automatic differentiation
- Understand the working mechanism of **SGD(momentum) optimizer**
- Learn to **drive development with unit tests**
- Understand memory management on different devices (CPU and CUDA)
- Understand the registration of operators and the basic flow of dispatching to different devices
- Get acquainted with **cmake** compilation
- Be Familiar with **pybind11** and the mechanism to export C++ source to Python
- Gain knowledge of the Python **pip** package management mechanism

## Compilation and Installation

clone:

`git clone git@github.com:yewentao256/MicroTorch.git`

Build MicroTorch:

```bash
pip install .
pip install . -v        # -v provides more details during installation
DEBUG=1 pip install .   # Compile the DEBUG version
CUDA=1 pip install .    # Compile the CUDA version

DEBUG=1 CUDA=1 pip install . -v
```

## Sample Program

Run the sample program using `python demo.py`. The sample program constructs and processes the following computational graph.

![image](resources/demo_graph.png)

## Project Structure

```bash
├── CMakeLists.txt
├── demo.py                     # Sample program
├── include                     # Directory for header files
├── microtorch
│   ├── __init__.py
│   ├── optimizer.py            # Optimizer encapsulation
│   ├── tensor.py               # Tensor encapsulation
│   ├── tests                   # Unit tests
│   │   ├── test_cpp.py
│   │   ├── test_cuda.py
│   │   ├── test_grad.py
│   │   ├── test_optimizer.py
│   │   └── test_tensor.py
│   └── utils.py                # Utility function encapsulation
├── pybind11-2.10               # Simplified pybind, export C++ to Python
├── pyproject.toml              # Package configuration
├── setup.py                    # Package installation file
└── src
    ├── binding                 # Interface for exporting C++ to Python
    ├── core                    # Core runtime components
    ├── cpu                     # CPU operators
    └── cuda                    # CUDA operators
```

For invocation flow, take `microtorch.sum()` as an example:

`microtorch.sum` -> `_microtorch.sum` -> `binding:sum` -> `core:ops` -> `sumOp(cpu/cuda)`

## Unit Tests

Conduct unit tests using `pytest`.

If you're learning PyTorch through MicroTorch, it's highly recommended to start by examining the unit tests. All unit tests are located in `microtorch/tests`.

## References

- [Pytorch](https://github.com/pytorch/pytorch)
- [Deep_dive_to_pytorch_autograd](https://wentao.site/deep_dive_to_autograd_1/)
- [Pytorch_under_the_hood](https://wentao.site/deep_dive_into_contiguous_1/)
- [TinyTorch](https://github.com/darglein/TinyTorch)
- [Pybind11](https://pybind11.readthedocs.io/en/stable/index.html)
- [Cmake_cpp_pybind11_tutorial](https://github.com/smrfeld/cmake_cpp_pybind11_tutorial)
- [Cuda_samples](https://github.com/NVIDIA/cuda-samples)
- [Simple-tensor](https://github.com/XuHQ1997/simple-tensor)
