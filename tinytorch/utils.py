import _tinytorch


def unit_test_cpp() -> None:
    return _tinytorch.unit_test()


def is_cuda_available() -> bool:
    return _tinytorch.is_cuda_available()
