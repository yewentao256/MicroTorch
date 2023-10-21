"""
Copyright (c) 2022-2023 yewentao
Licensed under the MIT License.
"""

import microtorch


def test_cuda() -> None:
    if not microtorch.cuda.is_cuda_available():
        return
    t = microtorch.rand([10, 20], "cuda")
    assert t.is_cuda()
    assert t.shape() == [10, 20]

    a = microtorch.ones(5).cuda()
    b = microtorch.Tensor([1, 2, 3, 4, 5], device='cuda')

    a.fill_(3)  # fill_
    assert a[3] == 3    # index_get

    assert (a + b)[0] == 4       # add

    assert (a - b)[1] == 1       # sub
    assert (a * b)[4] == 15      # mul
    assert (a * 3)[2] == 9       # mul scalar
    assert (
        a / b).equal(microtorch.Tensor([3, 1.5, 1, 0.75, 0.6], device="cuda"))

    c = microtorch.sum(microtorch.square(b))  # square and sum
    assert c[0] == 55       # 1+4+9+16+25 = 55

    d = c.clone()   # clone
    assert d[0] == 55

    f = a == b  # equal
    assert f.is_cuda()
    assert microtorch.sum(f)[0] == 1


def test_big_cuda_tensor():
    if not microtorch.cuda.is_cuda_available():
        return
    t = microtorch.rand([4, 1024, 1024, 1024], "cuda")  # 16GB tensor
    t.fill_(100)
    assert t[0, 0, 0, 0] == 100


if __name__ == "__main__":
    test_cuda()
    # test_big_cuda_tensor()
    print("successfully pass the test!")
