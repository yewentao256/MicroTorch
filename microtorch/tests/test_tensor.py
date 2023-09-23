import microtorch


def test_tensor_one_dimension() -> None:
    t1 = microtorch.ones(3)  # ones op
    t2 = microtorch.Tensor([1, 2, 3])  # init through list
    t3 = microtorch.zeros(3)  # zero op
    assert t2[2] == 3  # index get value
    assert (t1 + t2)[1] == 3  # add
    assert (t1 - t2)[1] == -1  # sub
    assert (t2 * t2)[2] == 9  # mult
    assert microtorch.sum(microtorch.square(t2))[0] == 14  # sum and square

    assert t3.is_contiguous()  # contiguous
    assert t1.zero_()[0] == 0

    t3[2] = 2       # set by index
    assert t3[2] == 2

    t4 = t3         # assign
    assert t4[2] == t3[2]

    t4.fill_(7.5)     # fill_
    assert t4[1] == 7.5

    assert t4.numel() == 3  # element count
    assert t4.shape()[0] == 3

    # In-place operations
    t4 += t4
    assert t4[2] == 15.0
    t4 -= t4
    assert t4[2] == 0.0

    # device
    assert t4.device() == "cpu"


def test_tensor_two_dimension() -> None:
    # 2D tensor creation
    t1 = microtorch.ones((2, 3))
    t2 = microtorch.ones((2, 3))
    t3 = microtorch.zeros((2, 3))

    # Indexing and value retrieval
    assert t2[1, 2] == 1

    # # Basic arithmetic operations
    assert (t1 + t2)[0, 1] == 2
    assert (t2 - t1)[0, 0] == 0
    assert (t1 * t2)[0, 2] == 1
    assert (t1 * t3)[1, 2] == 0

    assert t3.is_contiguous()
    assert t1.zero_()[1, 1] == 0

    # set by index
    t2[1, 2] = 7
    assert t2[1, 2] == 7

    assert t1.shape() == [2, 3]

    squared = microtorch.square(t2)
    assert squared[0, 0] == 1
    assert squared[1, 2] == 49

    summed = microtorch.sum(squared)
    assert summed[0] == 54  # 1*5 + 49
    

def test_big_tensor() -> None:
    t = microtorch.rand([1024, 1024, 64])   # tensor with 256 MB
    t.fill_(100)


if __name__ == "__main__":
    # test_tensor_one_dimension()
    # test_tensor_two_dimension()
    test_big_tensor()
    print("successfully pass the test!")
