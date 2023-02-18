import tinytorch
import time

def test_main():
    observation = tinytorch.rand(30).cuda()
    target = tinytorch.rand(30).cuda()
    print(f'observation: {observation}')
    print(f'target: {target}')
    params = []
    for i in range(4):
        params.append(tinytorch.rand(30).cuda())
        tinytorch.make_parameter(params[-1])

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


def test_cuda():
    a = tinytorch.rand(30000).cuda()
    b = tinytorch.rand(30000).cuda()
    print(a)
    now = time.time()
    c = a + b
    print(c)
    print(f"time usage: {time.time()- now}")

if __name__ == '__main__':
    test_main()
    test_cuda()