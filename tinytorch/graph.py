import tinytorch
import _tinytorch


def backward(tensor: tinytorch.Tensor) -> None:
    _tinytorch.backward(tensor)
