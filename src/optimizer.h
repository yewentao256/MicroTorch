#pragma once

#include "tensor.h"
#include "ops.h"

namespace tinytorch
{

    struct SGDOptimizer // stochastic gradient descent
    {
        int step = 0;
        float epsilon = 1e-6;
        float lr;
        float momentum = 0.9;
        std::vector<Tensor> params;

        SGDOptimizer(std::vector<Tensor> t_lst, float lr) : params(t_lst), lr(lr)
        {

        }

        void ZeroGrad()
        {
            for (Tensor &t : params)
            {
                t.clearGrad();
            }
        }

        void Step()
        {
            // update the weight of params
            for (size_t p = 0; p < params.size(); p++)
            {
                Tensor &param = params[p];
                if (param.grad().size() == 0)
                {
                    continue;
                }
                for (size_t i = 0; i < param.size(); i++)
                {
                    assert(param.grad().size() == param.size());
                    float g = param.grad()[i];

                    // sgd with nesterov momentum
                    // equation from pytorch
                    if (step > 0)
                    {
                        float b = momentum * b + 0.9 * g;
                        g = g + momentum * b;
                    }
                    param[i] = param[i] - lr * g;
                }
            }
            step++;
        }
    };

} // namespace tinytorch
