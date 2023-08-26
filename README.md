# MicroTorch

English | [简体中文](README.zh-cn.md)

**MicroTorch**: Demystifying the Magic of Pytorch

Have you ever been curious about the underlying principles of PyTorch? Or have you ever wanted to build an operator from scratch?

Welcome to MicroTorch!

Gone are the intimidating complexities. MicroTorch offers a simplified, intuitive implementation to get you started with the essence of deep learning. You will understand how computational graphs are constructed, how automatic differentiation is implemented, and possess the capabilities for basic CUDA operator development.

## Features

With MicroTorch, you can:

- Start from scratch and customize a complete Tensor class by yourself.
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

```python
# demo.py
import microtorch


def test_main(device: str = "cpu") -> None:
    observation = microtorch.rand(256, device, requires_grad=True)
    target = microtorch.rand(256, device)
    params = []
    for i in range(4):
        params.append(microtorch.rand(256, device, requires_grad=True))

    def model(x: microtorch.Tensor) -> microtorch.Tensor:
        x = x * params[0]
        x = x + params[1]
        x = x * params[2]
        x = x + params[3]
        return x

    # Create a simple optimizer
    optimizer = microtorch.SGDOptimizer(params, 0.1)

    # Optimize the model for 50 iterations
    for i in range(50):
        optimizer.zero_grad()
        prediction = model(observation)
        loss = microtorch.sum(microtorch.square(prediction - target))
        loss.backward()
        optimizer.step()
        print(f'Iter: {i}, Loss: {loss[0]}')


if __name__ == '__main__':
    device = "cuda" if microtorch.is_cuda_available() else "cpu"
    print(f"using `{device}` to test main")
    test_main(device)
```

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

Conduct unit tests using command `pytest`.

If you're learning PyTorch through MicroTorch, it's highly recommended to start by examining the unit tests. All unit tests are located in `microtorch/tests`.

For example:

```python
# microtorch/tests/test_grad.py

def test_autograd_2() -> None:
    x = Tensor([3.0], requires_grad=True)
    y = Tensor([4.0], requires_grad=True)

    a = x * y       # a = 3 * 4 = 12
    a.backward()
    assert x.grad()[0] == 4
    assert y.grad()[0] == 3

    # Reset gradients for the next test
    x.zero_grad()
    y.zero_grad()
    assert x.grad()[0] == 0.0
    assert y.grad()[0] == 0.0

    b = x + y       # b = 3 + 4 = 7
    assert b[0] == 7.0

    c = a * b       # c = 12 * 7 = 84
    assert c[0] == 84.0

    c.backward()    # dc/dx = y*b = 4*7 = 28 and dc/dy = x*b = 3*7 = 21

    assert x.grad()[0] == 40.0
    assert y.grad()[0] == 33.0

    x.zero_grad()
    y.zero_grad()

    # Using chain rule
    d = 5 * a + 2 * b
    assert d[0] == 74.0   # 5*12 + 2*7 = 60 + 14 = 74
    # dd/dx = 5*y + 2 = 5*4 + 2 = 22, dd/dy = 5*x + 2 = 5*3 + 2 = 17
    d.backward()

    # Checks for gradients after chain rule application
    assert x.grad()[0] == 22.0
    assert y.grad()[0] == 17.0
```

## References

- [Pytorch](https://github.com/pytorch/pytorch)
- [Deep_dive_to_pytorch_autograd](https://wentao.site/deep_dive_to_autograd_1/)
- [Pytorch_under_the_hood](https://wentao.site/deep_dive_into_contiguous_1/)
- [TinyTorch](https://github.com/darglein/TinyTorch)
- [Pybind11](https://pybind11.readthedocs.io/en/stable/index.html)
- [Cmake_cpp_pybind11_tutorial](https://github.com/smrfeld/cmake_cpp_pybind11_tutorial)
- [Cuda_samples](https://github.com/NVIDIA/cuda-samples)
- [Simple-tensor](https://github.com/XuHQ1997/simple-tensor)
