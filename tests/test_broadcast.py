"""
Copyright (c) 2022-2024 yewentao256
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

    # shape [3, 2] + [3, 1]
    t1 = microtorch.ones([3, 2], requires_grad=True)
    t2 = microtorch.ones([3, 1], requires_grad=True)
    t3 = t1 + t2
    assert t3.shape() == [3, 2]
    t4 = microtorch.sum(t3)
    t4.backward()
    assert t1.grad().shape() == [3, 2]
    assert t1.grad()[0, 0] == 1
    assert t2.grad().shape() == [3, 1]
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


def test_broadcast_cuda() -> None:
    if not microtorch.cuda.is_cuda_available():
        return
    # shape [3] + [1]
    t1 = microtorch.Tensor([1, 2, 3], requires_grad=True, device="cuda")
    t2 = microtorch.Tensor([1], requires_grad=True, device="cuda")
    t3 = t1 + t2
    assert t3.equal(microtorch.Tensor([2, 3, 4], device="cuda"))

    # TODO: backward CUDA supprot--sum_dim
    # t4 = microtorch.sum(t3, 0, True)
    # assert t4.shape() == [1]
    # assert t4.item() == 9
    # t4.backward()
    # assert t1.grad().equal(microtorch.Tensor([1, 1, 1], device="cuda"))
    # assert t2.grad().equal(microtorch.Tensor([3], device="cuda"))

    # shape [3, 2] + [3, 1]
    t1 = microtorch.ones([3, 2], requires_grad=True, device="cuda")
    t2 = microtorch.ones([3, 1], requires_grad=True, device="cuda")
    t3 = t1 + t2
    assert t3.shape() == [3, 2]
    t4 = microtorch.sum(t3)
    # TODO: backward CUDA supprot--sum_dim
    # t4.backward()
    # assert t1.grad().shape() == [3, 2]
    # assert t1.grad()[0, 0] == 1
    # assert t2.grad().shape() == [3, 1]
    # assert t2.grad()[0, 0] == t2.grad()[1, 0] == 2


if __name__ == "__main__":
    test_broadcast()
    test_different_dimensional_broadcast()
    test_broadcast_cuda()
    print("successfully pass the test!")
