#pragma once
#include "graph.hpp"

class AttentionHead
{
private:
    // Dimensions
    int d_in; // dimension of input vector
    int d_k;  // Dimension for Q and K (Similarity space)
    int d_v;  // Dimension for v( iNFORMATION Space)

    // NodeIDs for learnable weights 
    NodeID w_q; // shape: [d_in x d_k]
    NodeID w_k;  // shape: [d_in x d_k]
    NodeID w_v;  // shape: [d_in x d_k] 
public:
    // Constructor to initialise weightt
    AttentionHead(Graph& g, int in_dim, int k_dim, int v_dim);
    //forward pass
    NodeID forward(Graph& g, NodeID x_id);
};

class MultiHeadAttention
{
private:
    int d_model;
    int num_heads;
    std::vector<AttentionHead> heads;
    NodeID w_o; // output projection
    
    

public:
    MultiHeadAttention(Graph& g, int d_model, int num_heads);
    NodeID forward(Graph& g, NodeID x_id);

};