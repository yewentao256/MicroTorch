# MicroTorch

[English](README.md) | 简体中文

**MicroTorch**：揭开Pytorch的神秘面纱

你是否曾对PyTorch的底层原理感到好奇？或者是否曾想要自己动手，从零开始构建一个算子？

欢迎使用 MicroTorch！

不再是令人望而生畏的复杂框架，MicroTorch提供了一种简化、直观的实现，让你快速入门深度学习的核心。你将理解到计算图是如何构建的、自动微分是如何实现的，并拥有基础cuda算子开发的能力。

## 功能特性

通过 MicroTorch，你可以：

- 构建你自己的**tensor**类
- 理解基础算子前向、反向计算过程
- 理解构建计算图、自动微分的过程
- 理解**SGD(momentum)优化器**的工作机制
- 学会用**单元测试驱动开发**
- 学会在不同设备上管理内存（包括cpu和cuda）
- 了解算子的注册与dispatch到不同设备上的基本流程
- 学会使用**cmake**编译
- 了解**pybind11**和导出c++源码到python的机制
- 了解python **pip** 包管理机制

## 编译安装

clone:

`git clone git@github.com:yewentao256/MicroTorch.git`

Build MicroTorch:

```bash
pip install .
pip install . -v        # -v可以看到安装过程中更多信息
DEBUG=1 pip install .   # DEBUG环境变量编译debug版本
CUDA=1 pip install .    # 编译CUDA版本

DEBUG=1 CUDA=1 pip install . -v
```

## 示例程序

`python demo.py`运行示例程序

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

示例程序构建了一个如下的计算图进行运算

![image](resources/demo_graph.png)

## 项目结构

```bash
├── CMakeLists.txt
├── demo.py                     # 示例程序
├── include                     # 头文件目录
├── microtorch
│   ├── __init__.py
│   ├── optimizer.py            # optimizer封装
│   ├── tensor.py               # tensor封装
│   ├── tests                   # 单元测试
│   │   ├── test_cpp.py
│   │   ├── test_cuda.py
│   │   ├── test_grad.py
│   │   ├── test_optimizer.py
│   │   └── test_tensor.py
│   └── utils.py                # 辅助函数封装
├── pybind11-2.10               # 简化的pybind，用于导出C++内容到python
├── pyproject.toml              # 打包配置
├── setup.py                    # 打包安装文件
└── src
    ├── binding                 # 导出c++到python的接口
    ├── core                    # 核心运行组件
    ├── cpu                     # cpu算子
    └── cuda                    # cuda算子
```

调用流程，以`microtorch.sum()`为例：

`microtorch.sum` -> `_microtorch.sum` -> `binding:sum` -> `core:ops` -> `sumOp(cpu/cuda)`

## 单元测试

通过 `pytest` 进行单元测试

如果你正在通过MicroTorch学习pytorch，强烈推荐从单元测试开始看起。所有单元测试位于`microtorch/tests`

例如：

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

## 参考

- [Pytorch](https://github.com/pytorch/pytorch)
- [Deep_dive_to_pytorch_autograd](https://wentao.site/deep_dive_to_autograd_1/)
- [Pytorch_under_the_hood](https://wentao.site/deep_dive_into_contiguous_1/)
- [TinyTorch](https://github.com/darglein/TinyTorch)
- [Pybind11](https://pybind11.readthedocs.io/en/stable/index.html)
- [Cmake_cpp_pybind11_tutorial](https://github.com/smrfeld/cmake_cpp_pybind11_tutorial)
- [Cuda_samples](https://github.com/NVIDIA/cuda-samples)
- [Simple-tensor](https://github.com/XuHQ1997/simple-tensor)
