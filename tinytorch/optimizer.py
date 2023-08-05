import _tinytorch


class SGDOptimizer(_tinytorch.SGDOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
