"""
Copyright (c) 2022-2023 yewentao
Licensed under the MIT License.
"""

import _microtorch


class SGDOptimizer(_microtorch.SGDOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
