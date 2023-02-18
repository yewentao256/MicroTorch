import tinytorch


def test_tensor_one_dimension():
    t1 = tinytorch.ones(3)  # ones op
    t2 = tinytorch.Tensor([1, 2, 3])  # init through list
    t3 = tinytorch.zeros(3)  # zero op
    assert t2[2] == 3  # index get value

    # assert(t4.equal(tinytorch.Tensor([2,3,4])))    # TODO: equal
    assert (t1 + t2)[1] == 3  # add
    assert (t1 - t2)[1] == -1  # sub
    assert (t2 * t2)[2] == 9  # mult
    assert tinytorch.sum(tinytorch.square(t2))[0] == 14  # sum and square

    assert t3.is_contiguous()  # contiguous
    assert t2.size() == 3  # size


if __name__ == "__main__":
    test_tensor_one_dimension()
    print("successfully pass the test!")
