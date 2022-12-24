# TinyTorch

An Auto-Diff Optimization Framework for Teaching and Understanding Pytorch

## Building TinyTorch

```shell
cd TinyTorch
mkdir build && cd build
cmake ..
make
```

then you'll find a `.so` file, import it and begin to use in python!

## The Computational graph of main.cpp

![image](resources/TinyTorch_graph.png)

## Reference

- [TinyTorch](https://github.com/darglein/TinyTorch)
- [Pybind11](https://pybind11.readthedocs.io/en/stable/index.html)
- [Cmake_cpp_pybind11_tutorial](https://github.com/smrfeld/cmake_cpp_pybind11_tutorial)
- [Pytorch](https://github.com/pytorch/pytorch)