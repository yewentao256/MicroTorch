from __future__ import annotations
from typing import Union
import _tinytorch


class Tensor(_tinytorch.Tensor):
    def __init__(self, data, *args, **kwargs) -> None:
        if isinstance(data, _tinytorch.Tensor):
            super().__init__(data, *args, **kwargs)
        elif isinstance(data, (int, float)):
            super().__init__([data], *args, **kwargs)

        elif isinstance(data, (list, tuple)):
            if isinstance(data[0], (list, tuple)):
                raise ValueError(
                    "Unsupported multidimension data initialization now")
            else:
                super().__init__(data, *args, **kwargs)

        else:
            raise ValueError("Unsupported data type for Tensor initialization")

    def __repr__(self) -> str:
        return self.tensor_str()

    def __getitem__(self, index: Union[tuple, float]) -> float:
        if isinstance(index, tuple):
            return super().__getitem__(index)
        return super().__getitem__((index, ))

    def __setitem__(self, index: Union[tuple, float], value: float) -> None:
        if isinstance(index, tuple):
            super().__setitem__(index, value)
        else:
            super().__setitem__((index, ), value)

    def __add__(self, other: float) -> Tensor:
        return Tensor(super().__add__(other))

    def __mul__(self, other: float) -> Tensor:
        return Tensor(super().__mul__(other))

    def __sub__(self, other: float) -> Tensor:
        return Tensor(super().__sub__(other))

    def __iadd__(self, other: float) -> Tensor:
        return Tensor(super().__iadd__(other))

    def __isub__(self, other: float) -> Tensor:
        return Tensor(super().__isub__(other))

    def __imul__(self, other: float) -> Tensor:
        return Tensor(super().__imul__(other))
    
    def cpu(self) -> Tensor:
        return Tensor(super().cpu())

    def cuda(self) -> Tensor:
        return Tensor(super().cuda())


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
