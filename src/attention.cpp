#include "attention.hpp"
#include "init.hpp"

// Constructor
AttentionHead::AttentionHead(Graph& g, int in_dim, int k_dim, int v_dim)
    : d_in(in_dim), d_k(k_dim), d_v(v_dim)
{
    // Initialize weight matrices with Xavier initialization
    Tensor wq_tensor(d_in, d_k, 0.0);
    Tensor wk_tensor(d_in, d_k, 0.0);
    Tensor wv_tensor(d_in, d_v, 0.0);
    
    init::xavier_uniform(wq_tensor, d_in, d_k);
    init::xavier_uniform(wk_tensor, d_in, d_k);
    init::xavier_uniform(wv_tensor, d_in, d_v);
    
    w_q = g.parameter(wq_tensor);
    w_k = g.parameter(wk_tensor);
    w_v = g.parameter(wv_tensor);
}

// Ref:https://arxiv.org/pdf/1706.03762 (Attention is All you need 2017)
// Q @ K^T → scale → softmax → @ V → output
NodeID AttentionHead::forward(Graph& g, NodeID x_id)
{
    // 1. Linear Transforms
    // Project Input X into Q K V
    NodeID q = g.matmul(x_id, w_q);
    NodeID k = g.matmul(x_id, w_k);
    NodeID v = g.matmul(x_id, w_v);


    // 2 Scaled Dot Product
    // 2.1 -  Scores (Q* K^T)
    NodeID k_t = g.transpose(k);
    NodeID scores = g.matmul(q, k_t);

    // 3. scaling : we need to divide by sqrt(d_k)
    // Get the shape from scores tensor to create matching scale tensor
    const Tensor& scores_data = g.arena[scores].data;
    double scale_val = 1.0 / std::sqrt(static_cast<double>(d_k));
    NodeID scale_tensor = g.value(Tensor(scores_data.rows, scores_data.cols, scale_val));
    NodeID scaled_scores = g.mul(scores, scale_tensor);

    // 4. Softmax: get the weights
    NodeID attn_weights = g.softmax(scaled_scores);

    // 5. Value weighing : Weights * V
    NodeID output = g.matmul(attn_weights, v);

    return output;
}


// MultiHeadAttention COnstructor
MultiHeadAttention::MultiHeadAttention(Graph& g, int d_model_ , int num_heads_)
    : d_model(d_model_), num_heads(num_heads_)
{
    // each head has d_model / num_heads dimension
    int d_k = d_model / num_heads;
    int d_v = d_model / num_heads;

    // create heads
    for (int i =0; i < num_heads; ++i)
    {
        heads.emplace_back(g, d_model, d_k, d_v); // chcek notes for emplace explanation
    }

    //output projection

    Tensor wo_tensor(d_model, d_model, 0.0);
    init::xavier_uniform(wo_tensor, d_model, d_model);
    w_o = g.parameter(wo_tensor);

}

NodeID MultiHeadAttention::forward(Graph& g, NodeID x_id)
{
    // run all heads
    std::vector<NodeID> head_outputs;
    for (auto& head: heads)
    {
        head_outputs.push_back(head.forward(g,x_id));
    }

    //concat all heads
    NodeID concat = g.concat(head_outputs);

    // final output proj
    NodeID output = g.matmul(concat, w_o);

    return output;
}

//TODO: Implemenet a threading pool to allow multihead attention to run in parallel and not sequential 