import tinytorch


def test_cuda() -> None:
    if not tinytorch.is_cuda_available():
        return
    a = tinytorch.ones(5).cuda()
    b = tinytorch.Tensor([1, 2, 3, 4, 5]).cuda()
    assert a.is_cuda()
    
    a.fill_(3)  # fill_
    assert a[3] == 3    # index_get
    
    (a + b)[0] == 2     # add

    assert (a - b)[1] == 1    # sub
    assert (a * b)[4] == 15   # mul
    
    c = tinytorch.sum(tinytorch.square(b))  # square and sum
    assert c[0] == 55       # 1+4+9+16+25 = 55
    
 

if __name__ == "__main__":
    test_cuda()
    print("successfully pass the test!")
