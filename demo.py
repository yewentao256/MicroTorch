"""
Copyright (c) 2022-2024 yewentao256
Licensed under the MIT License.
"""

import microtorch


def test_main(device: str = "cpu") -> None:
    observation = microtorch.rand(256, device, requires_grad=True)
    target = microtorch.rand(256, device)
    params = []
    for i in range(4):
        params.append(microtorch.rand(256, device, requires_grad=True))

    def model(x: microtorch.Tensor) -> microtorch.Tensor:
        x = x * params[0]
        x = x + params[1]
        x = x * params[2]
        x = x + params[3]
        return x

    # Create a simple optimizer
    optimizer = microtorch.SGDOptimizer(params, 0.1)

    # Optimize the model for 50 iterations
    for i in range(50):
        optimizer.zero_grad()
        prediction = model(observation)
        loss = microtorch.sum(microtorch.square(prediction - target))
        loss.backward()
        optimizer.step()
        print(f"Iter: {i}, Loss: {loss[0]}")


if __name__ == "__main__":
    device = "cuda" if microtorch.cuda.is_cuda_available() else "cpu"
    print(f"using `{device}` to test main")
    test_main(device)
