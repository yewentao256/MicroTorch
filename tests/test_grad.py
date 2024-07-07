"""
Copyright (c) 2022-2024 yewentao256
Licensed under the MIT License.
"""

from microtorch import Tensor, AutoGradGuard


def test_autograd() -> None:
    x = Tensor([2.0], requires_grad=True)
    assert x.requires_grad

    y = x * 2
    assert y.item() == 4.0

    z = y.clone()  # clone
    assert z.item() == 4.0

    z.backward()
    # dz/dx = dy/dx = 2
    assert x.grad().item() == 2.0

    # chain rule
    zz = 2 * z
    assert zz.item() == 8.0
    zz.backward()  # z' = dz/dy * dy/dx = 2 * 2 = 4
    assert x.grad().item() == 6  # 6 = 4+2

    x.zero_grad()

    # Test negation
    w = -x
    assert w.item() == -2.0  # w = -x

    w.backward()
    # dw/dx = -1
    assert x.grad().item() == -1.0

    # Chain rule with negation
    ww = 3 * w
    assert ww.item() == -6.0
    ww.backward()  # ww' = d(-3x)/dx = -3
    assert x.grad().item() == -4.0  # -1 (previous grad) + -3 (new grad) = -4


def test_autograd_2() -> None:
    x = Tensor([3.0], requires_grad=True)
    y = Tensor([4.0], requires_grad=True)

    a = x * y  # a = 3 * 4 = 12
    b = x + y  # b = 3 + 4 = 7
    assert b.item() == 7.0
    c = a * b  # c = 12 * 7 = 84
    assert c.item() == 84.0
    d = 5 * a + 2 * b
    assert d.item() == 74.0  # 5*12 + 2*7 = 60 + 14 = 74

    # dd/dx = 5*y + 2 = 5*4 + 2 = 22, dd/dy = 5*x + 2 = 5*3 + 2 = 17
    d.backward()

    assert x.grad().item() == 22.0
    assert y.grad().item() == 17.0


def test_autograd_3() -> None:
    x = Tensor([2.0], requires_grad=True)
    y = Tensor([3.0], requires_grad=True)
    z = Tensor([4.0], requires_grad=True)

    # Compute some more complex expressions using x, y, and z
    a = x * y + z  # a = 2 * 3 + 4 = 10
    b = x * x + y * y  # b = 2^2 + 3^2 = 4 + 9 = 13
    c = a * b  # c = 10 * 13 = 130
    d = y * z + a  # d = 3 * 4 + 10 = 22
    e = c + d  # e = 130 + 22 = 152

    e.backward()  # de/dx, de/dy, de/dz

    # de/dx = 20 + 20 + 42 = 82
    # de/dy = 28 + 30 + 30 + 4 = 92
    # de/dz = 14 + 3 = 17
    assert x.grad().item() == 82.0
    assert y.grad().item() == 92.0
    assert z.grad().item() == 17.0


def test_autograd_mode() -> None:
    x = Tensor([2.0], requires_grad=True)
    with AutoGradGuard(False):
        y = x * 2
        y.backward()
        assert x.grad() is None


if __name__ == "__main__":
    test_autograd()
    test_autograd_2()
    test_autograd_3()
    test_autograd_mode()
    print("successfully pass the test!")
