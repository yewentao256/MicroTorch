import _tinytorch


class Tensor(_tinytorch.Tensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return self.tensor_str()


def zeros(size: list, device: str = 'cpu') -> Tensor:
    return Tensor(_tinytorch.zeros(size, device))


def ones(size: list, device: str = 'cpu') -> Tensor:
    return Tensor(_tinytorch.ones(size, device))


def rand(size: list, device: str = 'cpu') -> Tensor:
    return Tensor(_tinytorch.rand(size, device))


def sum(tensor: Tensor) -> float:
    return Tensor(_tinytorch.sum(tensor))


def square(tensor: Tensor) -> float:
    return Tensor(_tinytorch.square(tensor))
