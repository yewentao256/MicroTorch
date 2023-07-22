import _tinytorch


class Tensor(_tinytorch.Tensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return "repr tinytorch tensor!"


def rand(size, device='cpu'):
    cpp_tensor = _tinytorch.rand(size, device)
    print(type(cpp_tensor))
    print(cpp_tensor)
    return Tensor(cpp_tensor)