"""
Copyright (c) 2022-2023 yewentao
Licensed under the MIT License.
"""
import microtorch


def test_broadcast() -> None:
    # shape [3] + [1]
    t1 = microtorch.Tensor([1, 2, 3], requires_grad=True)
    t2 = microtorch.Tensor([1], requires_grad=True)
    t3 = t1 + t2
    assert t3.equal(microtorch.Tensor([2, 3, 4]))

    t4 = microtorch.sum(t3, 0, True)
    assert t4.shape() == [1]
    assert t4.item() == 9
    t4.backward()
    assert t1.grad().equal(microtorch.Tensor([1, 1, 1]))
    assert t2.grad().equal(microtorch.Tensor([3]))

    # shape [2, 2] + [2, 1]
    t1 = microtorch.ones([2, 2], requires_grad=True)
    t2 = microtorch.ones([2, 1], requires_grad=True)
    t3 = t1 + t2
    assert t3.shape() == [2, 2]
    t4 = microtorch.sum(t3)
    t4.backward()
    assert t1.grad().shape() == [2, 2]
    assert t1.grad()[0, 0] == 1
    assert t2.grad()[0, 0] == t2.grad()[1, 0] == 2


def test_different_dimensional_broadcast() -> None:
    t1 = microtorch.rand([3, 2, 2], requires_grad=True)
    t2 = microtorch.rand([2, 2], requires_grad=True)
    t3 = t1 + t2
    assert t3.shape() == [3, 2, 2]
    t4 = microtorch.sum(t3)
    assert t4.shape() == [1]
    assert t4.item() > 0
    t4.backward()
    assert t1.grad().shape() == [3, 2, 2]
    assert t2.grad().shape() == [2, 2]
    assert t2.grad()[0, 0] == 3


if __name__ == "__main__":
    test_broadcast()
    test_different_dimensional_broadcast()
    print("successfully pass the test!")
