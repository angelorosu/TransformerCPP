#pragma once
#include "graph.hpp"
#include "init.hpp"

class Layer
{
public:
    virtual ~Layer() = default;
    virtual NodeID forward(Graph& g, NodeID input) =0;
};

class Linear : public Layer
{
public:
    NodeID W, b;
    Linear(Graph& g, int in_features, int out_features);
    NodeID forward(Graph& g, NodeID input) override;
};