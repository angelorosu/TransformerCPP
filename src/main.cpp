#include <iostream>
#include "tensor.hpp"
#include "graph.hpp"
#include "optimizer.hpp"
#include "layers.hpp"
#include "loss.hpp"
#include "attention.hpp"

#include <cmath>



void test_linear_layer() {
    std::cout << "\n=== TEST: Linear Layer (y = 2x + 1) ===\n";
    
    Graph g;
    Linear layer(g, 1, 1);  // 1 input -> 1 output
    Adam optimizer(0.01);
    
    // Training data
    double X[] = {0, 1, 2, 3, 4};
    double Y[] = {1, 3, 5, 7, 9};
    
    for (int epoch = 0; epoch < 500; ++epoch) {
        double total_loss = 0.0;
        
        for (int i = 0; i < 5; ++i) {
            NodeID x = g.value(Tensor(1, 1, X[i]));
            NodeID target = g.value(Tensor(1, 1, Y[i]));
            
            NodeID pred = layer.forward(g, x);
            NodeID loss = Loss::mse(g, pred, target);
            
            total_loss += g.arena[loss].data.data[0];
            
            g.zero_grads();
            g.backward(loss);
            optimizer.step(g);
        }
        
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << total_loss / 5 << "\n";
        }
    }
    
    std::cout << "✓ Linear layer test complete!\n";
}

void test_attention_head() {
    std::cout << "\n=== TEST: Single Attention Head ===\n";
    
    Graph g;
    
    // Config
    std::size_t seq_len = 4;   // 4 tokens
    std::size_t d_model = 8;   // embedding dim
    std::size_t d_k = 8;       // key/query dim (same as d_model for single head)
    std::size_t d_v = 8;       // value dim

    
    // Create attention head
    AttentionHead attn(g, d_model, d_k, d_v);
    
    // Random input: (seq_len, d_model)
    Tensor input_data(seq_len, d_model, 0.0);
    for (std::size_t r = 0; r < seq_len; ++r) {
        for (std::size_t c = 0; c < d_model; ++c) {
            input_data.set(r, c, (double)(r * d_model + c) / (seq_len * d_model));
        }
    }
    
    NodeID x = g.value(input_data);
    
    // Forward pass
    std::cout << "Input shape: (" << seq_len << ", " << d_model << ")\n";
    
    NodeID output = attn.forward(g, x);
    
    const Tensor& out = g.arena[output].data;
    std::cout << "Output shape: (" << out.rows << ", " << out.cols << ")\n";
    
    // Print first few output values
    std::cout << "Output (first row): ";
    for (std::size_t c = 0; c < std::min((std::size_t)4, out.cols); ++c) {
        std::cout << out.get(0, c) << " ";
    }
    std::cout << "\n";
    
    // Test backward pass
    std::cout << "\nTesting backward pass...\n";
    
    // Create a dummy target and loss
    Tensor target_data(seq_len, d_k, 0.5);
    NodeID target = g.value(target_data);
    NodeID loss = Loss::mse(g, output, target);
    
    std::cout << "Loss: " << g.arena[loss].data.get(0, 0) << "\n";
    
    g.zero_grads();
    g.backward(loss);
    
    std::cout << "✓ Backward pass complete (no crash = success)\n";
    std::cout << "✓ Attention head test complete!\n";
}

void test_multihead_attention() {
    std::cout << "\n=== TEST: Multi-Head Attention ===\n";
    
    Graph g;
    
    std::size_t seq_len = 4;
    std::size_t d_model = 64;
    int num_heads = 4;
    
    MultiHeadAttention mha(g, d_model, num_heads);
    
    // Input: (seq_len, d_model)
    Tensor input_data(seq_len, d_model, 0.0);
    for (std::size_t r = 0; r < seq_len; ++r) {
        for (std::size_t c = 0; c < d_model; ++c) {
            input_data.set(r, c, (double)(r * d_model + c) / (seq_len * d_model));
        }
    }
    
    NodeID x = g.value(input_data);
    NodeID output = mha.forward(g, x);
    
    const Tensor& out = g.arena[output].data;
    std::cout << "Input shape: (" << seq_len << ", " << d_model << ")\n";
    std::cout << "Output shape: (" << out.rows << ", " << out.cols << ")\n";
    std::cout << "Num heads: " << num_heads << "\n";
    
    // Test backward
    Tensor target_data(seq_len, d_model, 0.5);
    NodeID target = g.value(target_data);
    NodeID loss = Loss::mse(g, output, target);
    
    std::cout << "Loss: " << g.arena[loss].data.get(0, 0) << "\n";
    
    g.zero_grads();
    g.backward(loss);
    
    std::cout << "✓ Multi-head attention test complete!\n";
}

void test_attention_learns() {
    std::cout << "\n=== TEST: Attention Actually Learns ===\n";
    
    Graph g;
    
    std::size_t seq_len = 4;
    std::size_t d_model = 16;
    
    AttentionHead attn(g, d_model, d_model, d_model);
    Adam optimizer(0.001);  // lower learning rate
    
    // Simpler input - smaller values
    Tensor input_data(seq_len, d_model, 0.0);
    for (std::size_t r = 0; r < seq_len; ++r) {
        for (std::size_t c = 0; c < d_model; ++c) {
            input_data.set(r, c, 0.01 * (r + c + 1));  // smaller values
        }
    }
    
    Tensor target_data = input_data;
    
    std::cout << "Task: Learn to reconstruct input\n";
    
    double initial_loss = 0.0;
    double final_loss = 0.0;
    
    // Track how many nodes are parameters (to preserve them)
    std::size_t num_params = g.arena.size();
    
    for (int epoch = 0; epoch < 50; ++epoch) {  // reduced epochs
        // Reset graph to just parameters (clear intermediate nodes)
        g.arena.resize(num_params);
        
        NodeID x = g.value(input_data);
        NodeID target = g.value(target_data);
        
        NodeID output = attn.forward(g, x);
        NodeID loss = Loss::mse(g, output, target);
        
        double loss_val = g.arena[loss].data.get(0, 0);
        
        if (std::isnan(loss_val) || std::isinf(loss_val)) {
            std::cout << "NaN/Inf detected at epoch " << epoch << "\n";
            
            // Debug: check weights
            std::cout << "Checking for bad values in graph...\n";
            for (std::size_t i = 0; i < g.arena.size(); ++i) {
                const Tensor& t = g.arena[i].data;
                for (std::size_t j = 0; j < t.size(); ++j) {
                    if (std::isnan(t.data[j]) || std::isinf(t.data[j])) {
                        std::cout << "  Bad value in node " << i << " (op=" << (int)g.arena[i].op << ")\n";
                        break;
                    }
                }
            }
            break;
        }
        
        if (epoch == 0) initial_loss = loss_val;
        final_loss = loss_val;
        
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << loss_val << "\n";
        }
        
        g.zero_grads();
        g.backward(loss);
        optimizer.step(g);
    }
    
    std::cout << "\nInitial loss: " << initial_loss << "\n";
    std::cout << "Final loss:   " << final_loss << "\n";
    
    if (final_loss < initial_loss * 0.5) {
        std::cout << "✓ Attention is learning!\n";
    } else {
        std::cout << "✗ Attention may not be learning properly\n";
    }
}

int main() {
    test_linear_layer();
    test_attention_head();
    test_multihead_attention();
    test_attention_learns();
    return 0;
}