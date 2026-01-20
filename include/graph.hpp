#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>
#include "node.hpp"

struct Graph
{
    std::vector<Node> arena;

    NodeID value(const Tensor& t);
    NodeID parameter(const Tensor& t);
    NodeID add(NodeID a_id, NodeID b_id);
    NodeID matmul(NodeID a_id, NodeID b_id);
    NodeID mul(NodeID a_id, NodeID b_id);
    NodeID div(NodeID a_id, NodeID b_id);
    


    NodeID relu(NodeID a_id);
    NodeID tanh(NodeID a_id);
    NodeID sigmoid(NodeID a_id);

    NodeID softmax(NodeID a_id);
    NodeID layer_norm(NodeID a_id);
    NodeID sqrt(NodeID a_id);
    NodeID mean(NodeID a_id);
    NodeID log(NodeID a_id); 
    NodeID transpose(NodeID a_id);
    NodeID concat(const std::vector<NodeID>& inputs);
    
    // TODO: Implement clear_intermediates() method
    // Problem: During training, each forward pass adds new nodes to the arena.
    //          After 500 epochs, arena grows to 25,000+ nodes → memory explodes.
    // Solution: After backward() and optimizer.step(), clear all non-parameter nodes.
    //          Parameters (weights) must be preserved, intermediates (activations, 
    //          gradients, temporary values) can be discarded.
    // Implementation:
    //   1. Track which nodes are parameters (is_parameter = true)
    //   2. Either: a) Store param count and resize(), or 
    //              b) Compact arena keeping only params, or
    //              c) Use separate storage for params vs intermediates
    // Current workaround: manual g.arena.resize(num_params) in training loop


    void zero_grads();
    void backward(NodeID root);



private:
    void build_topo(NodeID v, std::vector<NodeID>& topo, std::vector<std::uint8_t>& seen);

};
