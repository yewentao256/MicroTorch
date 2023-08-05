import tinytorch
import _tinytorch


def make_parameter(tensor: tinytorch.Tensor) -> tinytorch.Tensor:
    return _tinytorch.make_parameter(tensor)


def backward(tensor: tinytorch.Tensor) -> None:
    _tinytorch.backward(tensor)
