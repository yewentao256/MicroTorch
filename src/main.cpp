#include "tensor.h"
#include "ops.h"
#include "graph.h"
#include "backward.h"
#include "optimizer.h" 


int main()
{
    // tensor data
    tinytorch::Tensor observation = tinytorch::rand(10);
    tinytorch::Tensor target      = tinytorch::rand(10);
    std::cout << "observation: " << observation << std::endl;
    std::cout << "target: " << target << std::endl;

    // The parameters of the model
    std::vector<tinytorch::Tensor> params;
    for (size_t i = 0; i < 4; i++)
    {
        params.push_back(tinytorch::rand(10));
        MakeParameter(params.back());
        std::cout<< "param:" << params.back() << std::endl;
    }

    // The model itself
    auto model = [&](tinytorch::Tensor x) -> tinytorch::Tensor
    {
        x = x * params[0];
        x = x + params[1];
        x = x * params[2];
        x = x + params[3];
        return x;
    };

    // Create a simple optimizer
    tinytorch::SGDOptimizer optimizer(params, 0.1);

    // Optimize the model for 50 iterations
    for (size_t i = 0; i < 50; i++)
    {
        optimizer.ZeroGrad();   // clear the current grads

        auto prediction = model(observation);
        auto loss = sum(square(prediction - target));
        backward(loss);
        optimizer.Step();
        std::cout << "Step " << i << " Loss: " << loss << std::endl;
    }
    return 0;
}
