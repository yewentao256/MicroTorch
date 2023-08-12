import tinytorch


def test_tensor_one_dimension() -> None:
    t1 = tinytorch.ones(3)  # ones op
    t2 = tinytorch.Tensor([1, 2, 3])  # init through list
    t3 = tinytorch.zeros(3)  # zero op
    assert t2[2] == 3  # index get value
    assert (t1 + t2)[1] == 3  # add
    assert (t1 - t2)[1] == -1  # sub
    assert (t2 * t2)[2] == 9  # mult
    assert tinytorch.sum(tinytorch.square(t2))[0] == 14  # sum and square

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
    t1 = tinytorch.ones((2, 3))
    t2 = tinytorch.ones((2, 3))
    t3 = tinytorch.zeros((2, 3))

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

    squared = tinytorch.square(t2)
    assert squared[0, 0] == 1
    assert squared[1, 2] == 49

    summed = tinytorch.sum(squared)
    assert summed[0] == 54  # 1*5 + 49


if __name__ == "__main__":
    test_tensor_one_dimension()
    # test_tensor_two_dimension()
    print("successfully pass the test!")
