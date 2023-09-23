import microtorch


def test_cuda() -> None:
    if not microtorch.cuda.is_cuda_available():
        return
    t = microtorch.rand([10, 20], "cuda")
    assert t.is_cuda()
    assert t.shape() == [10, 20]

    a = microtorch.ones(5).cuda()
    b = microtorch.Tensor([1, 2, 3, 4, 5]).cuda()

    a.fill_(3)  # fill_
    assert a[3] == 3    # index_get

    (a + b)[0] == 2     # add

    assert (a - b)[1] == 1    # sub
    assert (a * b)[4] == 15   # mul

    c = microtorch.sum(microtorch.square(b))  # square and sum
    assert c[0] == 55       # 1+4+9+16+25 = 55

    d = c.clone()   # clone
    assert d[0] == 55


def test_big_cuda_tensor():
    if not microtorch.cuda.is_cuda_available():
        return
    # TODO: here [2, 1024, 1024, 1024] will cause a memory error
    t = microtorch.rand([1, 1024, 1024, 1024], "cuda")  # 4GB tensor
    t.fill_(100)


if __name__ == "__main__":
    test_cuda()
    test_big_cuda_tensor()
    print("successfully pass the test!")
