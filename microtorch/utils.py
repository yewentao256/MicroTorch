import _microtorch


def unit_test_cpp() -> None:
    return _microtorch.unit_test()


def is_cuda_available() -> bool:
    return _microtorch.is_cuda_available()
