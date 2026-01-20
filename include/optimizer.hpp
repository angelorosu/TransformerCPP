#pragma once
#include "graph.hpp"
#include <unordered_map>
//
class Optimizer
{
public:

    // Virtual destructor to ensure memory is cleaned up correctly
    virtual ~Optimizer() = default;
    //pass graph so optimizer can loop through whole arena
    virtual void step(Graph& g)=0;
};

// inherit from optimizer class
class SGD : public Optimizer
{
public: 
    double lr;
    //constructor to create SGD object
    SGD(double learning_rate);
    // iterate through graph and update weights
    void step(Graph& g) override;

};

class Adam : public Optimizer
{
public:
    double lr;
    double beta1 = 0.9; // decay for mean (momentum)
    double beta2; // decay for variance (scaling)
    double eps = 1e-8; 
    double max_grad_norm = 1.0;  // gradient clipping threshold
    int t = 0;  //time step

    //memory tensor Key: NodeID, Value
    std::unordered_map<NodeID, Tensor> m;
    std::unordered_map<NodeID, Tensor> v;


    Adam(double learning_rate);
    void step(Graph& g) override;

};


