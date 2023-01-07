import tinytorch

if __name__ == '__main__':
    observation = tinytorch.rand(30)
    target = tinytorch.rand(30)
    print(f'observation: {observation}')
    print(f'target: {target}')
    params = []
    for i in range(4):
        params.append(tinytorch.rand(30))
        tinytorch.make_parameter(params[-1])
        print(f'params: {params[-1]}')

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
        

    