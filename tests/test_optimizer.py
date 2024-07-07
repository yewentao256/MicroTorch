"""
Copyright (c) 2022-2024 yewentao256
Licensed under the MIT License.
"""

from microtorch import Tensor, ones, SGDOptimizer, sum, square
import microtorch


def test_one_iter(device="cpu") -> None:
    observation = Tensor([i for i in range(3)], device, requires_grad=True)
    target = ones(3, device)
    params = []
    for _ in range(4):
        params.append(ones(3, device, requires_grad=True))

    def model(x: Tensor) -> Tensor:
        x = x * params[0]
        x = x + params[1]
        x = x * params[2]
        x = x + params[3]
        return x

    # Create a simple optimizer
    optimizer = SGDOptimizer(params, 0.1)

    # Optimize the model for 1 iterations
    for _ in range(1):
        optimizer.zero_grad()
        prediction = model(observation)
        assert prediction.equal(Tensor([2, 3, 4]))
        loss = sum(square(prediction - target))
        assert loss[0] == 14  # 1 + 4 + 9
        loss.backward()
        optimizer.step()

    assert params[0].equal(Tensor([1.0, 0.6, -0.2]))
    assert params[0].grad().equal(Tensor([0.0, 4.0, 12.0]))
    assert params[1].equal(Tensor([0.8, 0.6, 0.4]))
    assert params[1].grad().equal(Tensor([2.0, 4.0, 6.0]))
    assert params[2].equal(Tensor([0.8, 0.2, -0.8]))
    assert params[2].grad().equal(Tensor([2.0, 8.0, 18.0]))
    assert params[3].equal(Tensor([0.8, 0.6, 0.4]))
    assert params[3].grad().equal(Tensor([2.0, 4.0, 6.0]))


def test_three_iter_momentum_nesterov(device="cpu") -> None:
    observation = Tensor([i for i in range(3)], device, requires_grad=True)
    target = ones(3, device)
    params = []
    for _ in range(2):
        params.append(ones(3, device, requires_grad=True))

    def model(x: Tensor) -> Tensor:
        x = x * params[0]
        x = x + params[1]
        return x

    # Create a simple optimizer
    optimizer = SGDOptimizer(params, 0.1, momentum=0.9)

    # Optimize the model for 3 iterations
    for _ in range(3):
        optimizer.zero_grad()
        prediction = model(observation)
        loss = sum(square(prediction - target))
        loss.backward()
        optimizer.step()
    assert params[0].equal(Tensor([1.0, 0.23, -0.448]))
    assert params[0].grad().equal(Tensor([0.0, 0.0, -7.2]))
    assert params[1].equal(Tensor([1.0, 0.23, 0.276]))
    assert params[1].grad().equal(Tensor([0.0, 0.0, -3.6]))


if __name__ == "__main__":
    test_one_iter("cpu")
    test_three_iter_momentum_nesterov("cpu")
    if microtorch.cuda.is_cuda_available():
        test_one_iter("cuda")
        test_three_iter_momentum_nesterov("cuda")
    print("successfully pass the test!")
