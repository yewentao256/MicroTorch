"""
Copyright (c) 2022-2023 yewentao
Licensed under the MIT License.
"""

import microtorch


def test_sum_with_dims_and_keepdim() -> None:
    t1 = microtorch.ones([2, 3], requires_grad=True)
    sum_t1 = microtorch.sum(t1, dims=0, keep_dim=False)
    assert sum_t1.equal(microtorch.Tensor([2, 2, 2]))

    sum_t2 = microtorch.sum(t1, dims=1, keep_dim=False)
    assert sum_t2.equal(microtorch.Tensor([3, 3]))

    sum_t3 = microtorch.sum(t1, dims=[0, 1], keep_dim=False)
    assert sum_t3.item() == 6.0

    sum_t4 = microtorch.sum(t1, dims=[0], keep_dim=True)
    assert sum_t4.shape() == [1, 3]
    assert sum_t4[0, 1] == 2

    sum_t5 = microtorch.sum(t1, dims=[1], keep_dim=True)
    assert sum_t5.shape() == [2, 1]
    assert sum_t5[1, 0] == 3


if __name__ == "__main__":
    test_sum_with_dims_and_keepdim()
    print("successfully pass the test!")
