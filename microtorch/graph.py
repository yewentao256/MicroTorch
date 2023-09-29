import _microtorch


class AutoGradGuard(_microtorch.AutoGradGuard):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
