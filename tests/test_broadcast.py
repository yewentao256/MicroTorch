"""
Copyright (c) 2022-2023 yewentao
Licensed under the MIT License.
"""
import microtorch


def test_broadcast_op() -> None:
    # shape [3] + [1]
    t1 = microtorch.Tensor([1, 2, 3], requires_grad=True)
    t2 = microtorch.Tensor([1], requires_grad=True)
    t3 = t1 + t2
    assert t3.equal(microtorch.Tensor([2, 3, 4]))

    t4 = microtorch.sum(t3)
    t4.backward()
    print(t1.grad())  # should be [1, 1, 1]
    print(t2.grad())  # [1] now, TODO: should be [3]

    t1 = microtorch.ones([2, 2])
    t2 = microtorch.ones([2, 1])
    t3 = t1 + t2
    assert microtorch.sum(t3)[0] == 8
    assert t3.shape() == [2, 2]


if __name__ == "__main__":
    test_broadcast_op()
    print("successfully pass the test!")
