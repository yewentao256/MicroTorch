"""
Copyright (c) 2022-2024 yewentao256
Licensed under the MIT License.
"""

import _microtorch


def is_cuda_available() -> bool:
    return _microtorch.is_cuda_available()


def synchronize() -> bool:
    return _microtorch.cuda_synchronize()
