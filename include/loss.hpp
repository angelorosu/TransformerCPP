#pragma once
#include "graph.hpp"

namespace Loss
{
//MSE -> 1/n * sum(pred - target)^2

inline NodeID mse(Graph& g, NodeID pred, NodeID target)
{
    // diff = pred - target
    NodeID neg_target = g.mul(target, g.value(Tensor(1,1,-1.0))); // TODO: MAKE A subtraction for g later
    NodeID diff = g.add(pred, neg_target);

    // square diff
    NodeID squared = g.mul(diff,diff);

    return squared;
}

inline NodeID cross_entropy(Graph& g, NodeID pred, NodeID target)
{   
    // log(pred)
    NodeID log_p = g.log(pred);
    // target * log(pred)
    NodeID product = g.mul(target, log_p);

    // negate : -(target * log(pred))
    return g.mul(product, g.value(Tensor(1,1,-1.0)));
}


}