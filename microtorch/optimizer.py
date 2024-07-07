"""
Copyright (c) 2022-2024 yewentao256
Licensed under the MIT License.
"""

import _microtorch


class SGDOptimizer(_microtorch.SGDOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
