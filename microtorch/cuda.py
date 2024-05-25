"""
Copyright (c) 2022-2023 yewentao
Licensed under the MIT License.
"""

import _microtorch


def is_cuda_available() -> bool:
    return _microtorch.is_cuda_available()


def synchronize() -> bool:
    return _microtorch.cuda_synchronize()
