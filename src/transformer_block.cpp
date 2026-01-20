#include "transformer_block.hpp"

//         x (input)
//         │
//         ▼
//    ┌─────────┐
//    │LayerNorm│
//    └────┬────┘
//         │
//         ▼
//    ┌─────────┐
//    │  MHA    │  (Multi-Head Attention)
//    └────┬────┘
//         │
//         ▼
//    ┌─────────┐
//    │   +     │ ◄─── x (residual connection)
//    └────┬────┘
//         │
//         ▼
//    ┌─────────┐
//    │LayerNorm│
//    └────┬────┘
//         │
//         ▼
//    ┌─────────┐
//    │  FFN    │  (ff1 → ReLU → ff2)
//    └────┬────┘
//         │
//         ▼
//    ┌─────────┐
//    │   +     │ ◄─── residual1 (residual connection)
//    └────┬────┘
//         │
//         ▼
//       output


EncoderBlock::EncoderBlock(Graph& g, const TransformerConfig& cfg)
    : config(cfg),
    mha(g,cfg.d_model, cfg.num_heads),
    ff1(g,cfg.d_model, cfg.num_heads),
    ff2(g,cfg.d_ff, cfg.num_heads)
{

}


NodeID EncoderBlock::forward(Graph& g, NodeID x_id)
{
    // Pre-LN style (GPT-2, modern transformers)
    
    // 1. LayerNorm → MHA → Residual
    NodeID norm1 = g.layer_norm(x_id);
    NodeID attn_out = mha.forward(g,norm1);
    NodeID residual1 = g.add(x_id, attn_out);

    // 2. LayerNorm -> FFN -> Residual
    NodeID norm2 = g.layer_norm(residual1);
    NodeID ff_hidden = ff1.forward(g,norm2);
    NodeID ff_act = g.relu(ff_hidden);
    NodeID ff_out = ff2.forward(g, ff_act);
    NodeID residual2 = g.add(residual1, ff_out);

    return residual2;



}


