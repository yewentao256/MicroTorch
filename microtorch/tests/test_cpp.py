import microtorch


def test_cpp_unit_test() -> None:
    assert microtorch.unit_test_cpp()


if __name__ == '__main__':
    test_autograd()
    test_cpp_unit_test()
    print('successfully pass the test!')