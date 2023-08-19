import tinytorch


def test_main(device: str = "cpu") -> None:
    observation = tinytorch.rand(30, device, requires_grad=True)
    target = tinytorch.rand(30, device)
    params = []
    for i in range(4):
        params.append(tinytorch.rand(30, device, requires_grad=True))

    def model(x: tinytorch.Tensor) -> tinytorch.Tensor:

        x = x * params[0]
        x = x + params[1]
        x = x * params[2]
        x = x + params[3]
        return x

    # Create a simple optimizer
    optimizer = tinytorch.SGDOptimizer(params, 0.1)

    # Optimize the model for 50 iterations
    for i in range(50):
        optimizer.zero_grad()
        prediction = model(observation)
        loss = tinytorch.sum(tinytorch.square(prediction - target))
        tinytorch.backward(loss)
        optimizer.step()
        print(f'Iter: {i}, Loss: {loss[0]}')


if __name__ == '__main__':
    device = "cuda" if tinytorch.is_cuda_available() else "cpu"
    print(f"using `{device}` to test main")
    test_main(device)
