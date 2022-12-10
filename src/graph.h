#pragma once

#include "tensor.h"

#include <map>

namespace tinytorch
{

struct Node;

struct Edge{
    /// The function this `Edge` points to.
    std::shared_ptr<Node> function;
    /// The identifier of a particular input to the function.
    uint32_t identifier;

    Edge(std::shared_ptr<Node> function, uint32_t identifier) noexcept : 
    function(std::move(function)), identifier(identifier){}
};

struct Context
{
    std::map<std::string, Tensor> data;
    std::map<std::string, int> data_int;
};

struct Node{
    // A global counter to get correct node ordering
    int sequence_number;
    // Inline variable, see https://stackoverflow.com/questions/38043442/how-do-inline-variables-work
    // Here we use it for unique increasing sequence number
    inline static int current_seq_nr; 

    // The next edges are the inputs of the forward operator
    std::vector<std::shared_ptr<Edge>> next;

    // Variables that are required for the backward pass
    Context context;

    int num_inputs;

    // Create a node and give it a unique increasing sequence number
    Node() : sequence_number(current_seq_nr++) {}

    // Computes and returns the gradients of the input tensor of the forward operator.
    // The input is the gradient of the forward output
    virtual std::vector<Tensor> backward(std::vector<Tensor> forward_output) = 0;
};

template <typename T>
struct FunctionNode : public Node{
    FunctionNode() {}
    static std::vector<Tensor> forward_and_build_graph(std::vector<Tensor> t_list) {
        // Create node and set next edge
        auto node = std::make_shared<FunctionNode<T>>();
        for (size_t i = 0; i < t_list.size(); i++)
        {
            // Here we bind the edge of tensor before to the current node
            (*node).next.push_back(t_list[i].getEdge());
        }
        (*node).num_inputs = t_list.size();

        // forward
        auto result = T::forward((*node).context, t_list);

        // Set the edges of the output to point to this node
        for (size_t i = 0; i < result.size(); i++)
        {
            result[i].setEdge(std::make_shared<Edge>(node, i));
        }
        return result;
    }

    std::vector<Tensor> backward(std::vector<Tensor> forward_output) override {
        auto grad_list = T::backward(context, forward_output);
        return grad_list;
    }
};

struct AccumulateGrad : public Node{
    // Each AccumulateGrad owns a tensor for calculating grad
    // Usually for updating params
    Tensor t;
    AccumulateGrad(Tensor t) : t(t) { num_inputs = 1; }
    std::vector<Tensor> backward(std::vector<Tensor> input_grad) override {
        assert(input_grad.size()==1);
        t.addGradInplace(input_grad[0]);
        return {};
    }
};

inline void MakeParameter(Tensor t)
{
    t.setEdge(std::make_shared<Edge>(std::make_shared<AccumulateGrad>(t), 0));
}

} // namespace tinytorch