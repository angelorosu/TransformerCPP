#pragma once
#include "graph.hpp"
#include "attention.hpp"
#include "layers.hpp"


struct TransformerConfig
{
    int d_model = 64; // embeddign dim
    int num_heads = 4; // attention heads( d_ model must be divisible by this though)
    int d_ff = 256; // feedfroward hidden dim
    //double droput - 0.0; TODO: add this later
};

class EncoderBlock
{
private:
    TransformerConfig config;
    MultiHeadAttention mha;
    Linear ff1;
    Linear ff2;

public:
    EncoderBlock(Graph& g, const TransformerConfig& cfg);
    NodeID forward(Graph& g, NodeID x_id);
};


//TODO: DecoderBlock Class


