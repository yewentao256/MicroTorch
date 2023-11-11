"""
Copyright (c) 2022-2023 yewentao
Licensed under the MIT License.
"""
from __future__ import annotations
from typing import Union
import _microtorch


class Tensor(_microtorch.Tensor):
    def __init__(self, data, *args, **kwargs) -> None:
        if isinstance(data, _microtorch.Tensor):
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
            raise ValueError(
                f"Unsupported type {type(data)} for Tensor initialization")

    def __repr__(self) -> str:
        return self.str()

    def __getitem__(self, index: Union[tuple, float]) -> float:
        if isinstance(index, tuple):
            return super().__getitem__(index)
        return super().__getitem__((index, ))

    def __setitem__(self, index: Union[tuple, float], value: float) -> None:
        if isinstance(index, tuple):
            super().__setitem__(index, value)
        else:
            super().__setitem__((index, ), value)

    def __add__(self, other: Union[Tensor, int, float]) -> Tensor:
        if isinstance(other, Tensor):
            return Tensor(super().__add__(other))
        return Tensor(super().__add__(Tensor(other)))

    def __sub__(self, other: Union[Tensor, int, float]) -> Tensor:
        if isinstance(other, Tensor):
            return Tensor(super().__sub__(other))
        return Tensor(super().__sub__(Tensor(other)))

    def __mul__(self, other: Union[Tensor, int, float]) -> Tensor:
        return Tensor(super().__mul__(other))

    def __truediv__(self, other: Union[Tensor, int, float]) -> Tensor:
        if isinstance(other, Tensor):
            return Tensor(super().__truediv__(other))
        return Tensor(super().__truediv__(Tensor(other)))

    def __iadd__(self, other: Union[Tensor, int, float]) -> Tensor:
        if isinstance(other, Tensor):
            return Tensor(super().__iadd__(other))
        return Tensor(super().__iadd__(Tensor(other)))

    def __isub__(self, other: Union[Tensor, int, float]) -> Tensor:
        if isinstance(other, Tensor):
            return Tensor(super().__isub__(other))
        return Tensor(super().__isub__(Tensor(other)))

    def __imul__(self, other: Union[Tensor, int, float]) -> Tensor:
        return Tensor(super().__imul__(other))

    def __itruediv__(self, other: Union[Tensor, int, float]) -> Tensor:
        if isinstance(other, Tensor):
            return Tensor(super().__itruediv__(other))
        return Tensor(super().__itruediv__(Tensor(other)))

    def __radd__(self, other: Union[Tensor, int, float]) -> Tensor:
        return self + other

    def __rsub__(self, other: Union[Tensor, int, float]) -> Tensor:
        return self - other

    def __rmul__(self, other: Union[Tensor, int, float]) -> Tensor:
        return self * other

    def __rtruediv__(self, other: Union[Tensor, int, float]) -> Tensor:
        return self / other

    def cpu(self) -> Tensor:
        return Tensor(super().cpu())

    def cuda(self) -> Tensor:
        return Tensor(super().cuda())

    def grad(self) -> Tensor:
        g = Tensor(super().grad())
        return g if g.defined() else None

    def clone(self) -> Tensor:
        return Tensor(super().clone())

    def __eq__(self, other: Tensor) -> bool:
        if isinstance(other, Tensor):
            return Tensor(super().__eq__(other))
        raise RuntimeError(f"unexpected compare type: {type(other)}")

    def defined(self) -> bool:
        return super().defined()

    def square(self) -> Tensor:
        return Tensor(super().square())


def _wrap_scalar_to_list(obj: Union[list, int, float]) -> list:
    if isinstance(obj, (int, float)):
        return [obj]
    return obj


def empty(size: Union[list, int, float], device: str = 'cpu',
          requires_grad: bool = False) -> Tensor:
    return Tensor(_microtorch.empty(
        _wrap_scalar_to_list(size), device, requires_grad))


def zeros(size: Union[list, int, float], device: str = 'cpu',
          requires_grad: bool = False) -> Tensor:
    return Tensor(_microtorch.zeros(
        _wrap_scalar_to_list(size), device, requires_grad))


def ones(size: Union[list, int, float], device: str = 'cpu',
         requires_grad: bool = False) -> Tensor:
    return Tensor(_microtorch.ones(
        _wrap_scalar_to_list(size), device, requires_grad))


def rand(size: Union[list, int, float], device: str = 'cpu',
         requires_grad: bool = False) -> Tensor:
    return Tensor(_microtorch.rand(
        _wrap_scalar_to_list(size), device, requires_grad))


def sum(tensor: Tensor, dims: Union[list[int], int, None] = None,
        keep_dim: bool = False) -> Tensor:
    if dims is None:
        return Tensor(_microtorch.sum(tensor))
    else:
        if not isinstance(dims, list):
            dims = [dims]
        return Tensor(_microtorch.sum_dim(tensor, dims, keep_dim))


def square(tensor: Tensor) -> Tensor:
    return Tensor(_microtorch.square(tensor))
