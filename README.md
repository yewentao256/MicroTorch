# TinyTorch

An Auto-Diff Optimization Framework for Teaching and Understanding Pytorch

## Through This Project, You Can

- Understand auto-diff and backpropagation
- Know how to use `pybind11`
- Get the idea of `cmake`, `pip` and package installation.

## Building TinyTorch

- clone the repository:

`git clone git@github.com:yewentao256/TinyTorch.git --recursive`

Note: `--recursive` is needed if you don't have `pybind11` installed correctly

- Build Tinytorch

```bash
pip install .
pip install . -v        # to see more information about installation
DEBUG=1 pip install .   # to install the debug version(in order to use gdb/lldb)
CUDA=1 pip install .    # to install the cuda version

DEBUG=1 CUDA=1 pip install . -v
```

## The Computational graph of main.cpp

![image](resources/TinyTorch_graph.png)

## Reference

- [TinyTorch](https://github.com/darglein/TinyTorch)
- [Pybind11](https://pybind11.readthedocs.io/en/stable/index.html)
- [Cmake_cpp_pybind11_tutorial](https://github.com/smrfeld/cmake_cpp_pybind11_tutorial)
- [Pytorch](https://github.com/pytorch/pytorch)
- [Cuda_samples](https://github.com/NVIDIA/cuda-samples)
- [Simple-tensor](https://github.com/XuHQ1997/simple-tensor)
