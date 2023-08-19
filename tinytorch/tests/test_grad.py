from tinytorch import Tensor


def test_autograd() -> None:
    x = Tensor([2.0], requires_grad=True)
    assert x.requires_grad

    y = x * 2
    assert y[0] == 4.0
    y.backward()
    # dy/dx = 2
    assert x.grad()[0] == 2.0

    # chain rule
    z = 2 * y
    assert z[0] == 8.0
    z.backward()        # z' = dz/dy * dy/dx = 2 * 2 = 4
    assert x.grad()[0] == 6    # 6 = 4+2


def test_autograd_2() -> None:
    x = Tensor([3.0], requires_grad=True)
    y = Tensor([4.0], requires_grad=True)

    a = x * y       # a = 3 * 4 = 12
    # a.backward()
    # assert x.grad()[0] == 4
    # assert y.grad()[0] == 3

    # # Reset gradients for the next test
    # x.zero_grad()
    # y.zero_grad()
    # assert x.grad()[0] == 0.0
    # assert y.grad()[0] == 0.0

    b = x + y       # b = 3 + 4 = 7
    assert b[0] == 7.0

    c = a * b       # c = 12 * 7 = 84
    assert c[0] == 84.0

    c.backward()    # dc/dx = y*b = 4*7 = 28 and dc/dy = x*b = 3*7 = 21

    print(x.grad())
    print(y.grad())
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


if __name__ == "__main__":
    test_autograd()
    test_autograd_2()
    print("successfully pass the test!")
