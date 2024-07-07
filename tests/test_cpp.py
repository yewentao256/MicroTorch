"""
Copyright (c) 2022-2024 yewentao256
Licensed under the MIT License.
"""

import _microtorch


def test_cpp_unit_test() -> None:
    assert _microtorch.unit_test()


if __name__ == "__main__":
    test_cpp_unit_test()
    print("successfully pass the test!")
