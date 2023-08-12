import tinytorch


def test_cuda() -> None:
    if not tinytorch.is_cuda_available():
        return
    a = tinytorch.ones(5).cuda()
    b = tinytorch.Tensor([1, 2, 3, 4, 5]).cuda()
    assert a.is_cuda()
    c = (a + b).cpu()   # add
    assert c[0] == 2
    
    a.fill_(3)  # fill_
    assert a.cpu()[3] == 3
    
    d = (a - b).cpu()         # sub
    assert d[1] == 1
    
    e = (a * b).cpu()         # mul
    assert e[4] == 15
    
    f = tinytorch.sum(tinytorch.square(b)).cpu()  # square and sum
    assert f[0] == 55       # 1+4+9+16+25 = 55
    
 

if __name__ == "__main__":
    test_cuda()
    print("successfully pass the test!")
