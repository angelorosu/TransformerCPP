#include <iostream>
#include "tensor.hpp"
#include "graph.hpp"
#include "optimizer.hpp"
#include "layers.hpp"
#include "loss.hpp"
#include "attention.hpp"
#include "positional_encoding.hpp"
#include "transformer_block.hpp"  // add this

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

void test_positional_encoding() {
    std::cout << "\n=== TEST: Positional Encoding ===\n";
    
    std::size_t seq_len = 4;
    std::size_t d_model = 8;
    
    Tensor pe = PE::sinusoidal(seq_len, d_model);
    
    std::cout << "Shape: (" << pe.rows << ", " << pe.cols << ")\n";
    std::cout << "PE matrix:\n";
    
    for (std::size_t r = 0; r < seq_len; ++r) {
        std::cout << "  pos " << r << ": ";
        for (std::size_t c = 0; c < d_model; ++c) {
            std::cout << pe.get(r, c) << " ";
        }
        std::cout << "\n";
    }
    
    // Check: pos 0 should be [0, 1, 0, 1, ...] (sin(0)=0, cos(0)=1)
    bool valid = (std::abs(pe.get(0, 0)) < 0.001) && (std::abs(pe.get(0, 1) - 1.0) < 0.001);
    
    if (valid) {
        std::cout << "✓ Position 0 encoding correct (sin(0)=0, cos(0)=1)\n";
    } else {
        std::cout << "✗ Position 0 encoding wrong\n";
    }
}

void test_encoder_block() {
    std::cout << "\n=== TEST: Encoder Block ===\n";
    
    Graph g;
    
    TransformerConfig cfg;
    cfg.d_model = 64;
    cfg.num_heads = 4;
    cfg.d_ff = 256;
    
    EncoderBlock encoder(g, cfg);
    
    std::size_t seq_len = 4;
    
    // Input: (seq_len, d_model)
    Tensor input_data(seq_len, cfg.d_model, 0.0);
    for (std::size_t r = 0; r < seq_len; ++r) {
        for (std::size_t c = 0; c < (std::size_t)cfg.d_model; ++c) {
            input_data.set(r, c, 0.01 * (r + c + 1));
        }
    }
    
    // Add positional encoding
    Tensor pe = PE::sinusoidal(seq_len, cfg.d_model);
    for (std::size_t r = 0; r < seq_len; ++r) {
        for (std::size_t c = 0; c < (std::size_t)cfg.d_model; ++c) {
            input_data.set(r, c, input_data.get(r, c) + pe.get(r, c));
        }
    }
    
    NodeID x = g.value(input_data);
    
    std::cout << "Input shape: (" << seq_len << ", " << cfg.d_model << ")\n";
    std::cout << "Config: d_model=" << cfg.d_model << ", heads=" << cfg.num_heads << ", d_ff=" << cfg.d_ff << "\n";
    
    // Forward pass
    NodeID output = encoder.forward(g, x);
    
    const Tensor& out = g.arena[output].data;
    std::cout << "Output shape: (" << out.rows << ", " << out.cols << ")\n";
    
    // Check for NaN
    bool has_nan = false;
    for (std::size_t i = 0; i < out.size(); ++i) {
        if (std::isnan(out.data[i])) {
            has_nan = true;
            break;
        }
    }
    
    if (has_nan) {
        std::cout << "✗ Output contains NaN\n";
        return;
    }
    
    std::cout << "Output (first row): ";
    for (std::size_t c = 0; c < std::min((std::size_t)4, out.cols); ++c) {
        std::cout << out.get(0, c) << " ";
    }
    std::cout << "...\n";
    
    // Test backward
    Tensor target_data(seq_len, cfg.d_model, 0.5);
    NodeID target = g.value(target_data);
    NodeID loss = Loss::mse(g, output, target);
    
    std::cout << "Loss: " << g.arena[loss].data.get(0, 0) << "\n";
    
    g.zero_grads();
    g.backward(loss);
    
    std::cout << "✓ Encoder block test complete!\n";
}

void test_encoder_learns() {
    std::cout << "\n=== TEST: Encoder Learns to Predict Next Number ===\n";
    
    Graph g;
    
    TransformerConfig cfg;
    cfg.d_model = 32;
    cfg.num_heads = 4;
    cfg.d_ff = 128;
    
    EncoderBlock encoder(g, cfg);
    Linear output_proj(g, cfg.d_model, 1);
    Adam optimizer(0.0005);  // Lower learning rate for stability
    
    std::size_t seq_len = 4;
    
    std::vector<std::vector<double>> inputs = {
        {1, 2, 3, 4},
        {2, 3, 4, 5},
        {3, 4, 5, 6},
        {5, 6, 7, 8},
        {10, 11, 12, 13}
    };
    
    std::vector<std::vector<double>> targets = {
        {2, 3, 4, 5},
        {3, 4, 5, 6},
        {4, 5, 6, 7},
        {6, 7, 8, 9},
        {11, 12, 13, 14}
    };
    
    std::size_t num_params = g.arena.size();
    
    std::cout << "Task: [1,2,3,4] → [2,3,4,5] (predict next number)\n\n";
    
    double initial_loss = 0.0;
    double final_loss = 0.0;
    
    for (int epoch = 0; epoch < 200; ++epoch) {
        double total_loss = 0.0;
        
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            g.arena.resize(num_params);
            
            Tensor input_data(seq_len, cfg.d_model, 0.0);
            for (std::size_t t = 0; t < seq_len; ++t) {
                input_data.set(t, 0, inputs[i][t] / 10.0);
            }
            
            Tensor pe = PE::sinusoidal(seq_len, cfg.d_model);
            for (std::size_t r = 0; r < seq_len; ++r) {
                for (std::size_t c = 0; c < (std::size_t)cfg.d_model; ++c) {
                    input_data.set(r, c, input_data.get(r, c) + pe.get(r, c));
                }
            }
            
            Tensor target_data(seq_len, 1, 0.0);
            for (std::size_t t = 0; t < seq_len; ++t) {
                target_data.set(t, 0, targets[i][t] / 10.0);
            }
            
            NodeID x = g.value(input_data);
            NodeID target = g.value(target_data);
            
            NodeID encoded = encoder.forward(g, x);
            NodeID pred = output_proj.forward(g, encoded);
            NodeID loss = Loss::mse(g, pred, target);
            
            double loss_val = g.arena[loss].data.get(0, 0);
            total_loss += loss_val;
            
            g.zero_grads();
            g.backward(loss);
            optimizer.step(g);
        }
        
        double avg_loss = total_loss / inputs.size();
        
        if (epoch == 0) initial_loss = avg_loss;
        final_loss = avg_loss;
        
        if (epoch % 50 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << avg_loss << "\n";
            
            // Print a prediction during training to debug
            g.arena.resize(num_params);
            Tensor test_in(seq_len, cfg.d_model, 0.0);
            for (std::size_t t = 0; t < seq_len; ++t) {
                test_in.set(t, 0, inputs[0][t] / 10.0);
            }
            Tensor pe_test = PE::sinusoidal(seq_len, cfg.d_model);
            for (std::size_t r = 0; r < seq_len; ++r) {
                for (std::size_t c = 0; c < (std::size_t)cfg.d_model; ++c) {
                    test_in.set(r, c, test_in.get(r, c) + pe_test.get(r, c));
                }
            }
            NodeID tx = g.value(test_in);
            NodeID te = encoder.forward(g, tx);
            NodeID tp = output_proj.forward(g, te);
            const Tensor& tout = g.arena[tp].data;
            std::cout << "  Pred: [";
            for (std::size_t t = 0; t < seq_len; ++t) {
                std::cout << tout.get(t, 0) * 10.0;
                if (t < seq_len - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
    }
    
    std::cout << "\nInitial loss: " << initial_loss << "\n";
    std::cout << "Final loss:   " << final_loss << "\n";
    
    if (final_loss < initial_loss * 0.3) {
        std::cout << "✓ Encoder is learning!\n";
    } else {
        std::cout << "✗ Encoder may not be learning properly\n";
    }
}

int main() {
    test_linear_layer();
    test_attention_head();
    test_multihead_attention();
    test_attention_learns();
    test_positional_encoding();
    test_encoder_block();
    test_encoder_learns();  // add this
    return 0;
}