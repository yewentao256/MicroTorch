import _microtorch


class SGDOptimizer(_microtorch.SGDOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
