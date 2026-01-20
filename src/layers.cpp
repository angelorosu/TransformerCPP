#include "layers.hpp"

Linear::Linear(Graph& g, int in_features, int out_features)
{
    // create weight tensor + initialse
    Tensor W_init(in_features,out_features,0.0);
    init::xavier_uniform(W_init, in_features,out_features);

    // make it paramerr in the graph
    W = g.parameter(W_init);

    // Bias starting at 0.0
    Tensor b_init(1,out_features,0.0);
    b = g.parameter(b_init);
}

NodeID Linear::forward(Graph& g, NodeID input)
{
    NodeID xW = g.matmul(input,W);
    return g.add(xW,b);
}