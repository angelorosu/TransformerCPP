# TransformerCPP

A Transformer neural network built from scratch in C++17, with custom autograd engine.
> ⚠️ **Note:** This project is under active development. Building towards a complete Transformer implementation with parallel multi-head attention.

<p align="center">
  <img src="https://img.shields.io/badge/C++-17-blue.svg" alt="C++17">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT">
  <img src="https://img.shields.io/badge/Build-CMake-red.svg" alt="CMake">
  <img src="https://img.shields.io/badge/Threading-Optional-orange.svg" alt="Threading">
</p>

## What is This?

A from-scratch implementation of the Transformer architecture ("Attention Is All You Need", 2017) including:
- Custom automatic differentiation engine
- Tensor operations with backpropagation  
- Attention mechanism with learnable Q, K, V projections
- Optional multi-threading for parallel attention heads

```
Input → [Multi-Head Attention] → [Add & Norm] → [FFN] → [Add & Norm] → Output
              ↓
        ┌─────┴─────┐
        │  N Heads  │  ← Parallel execution (optional)
        │  Q, K, V  │
        │  Softmax  │
        └───────────┘
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    TransformerCPP                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐           │
│   │ Tensor  │────▶│  Graph  │────▶│  Node   │           │
│   │         │     │         │     │  + Op   │           │
│   └─────────┘     └────┬────┘     └─────────┘           │
│                        │                                 │
│        ┌───────────────┼───────────────┐                │
│        ▼               ▼               ▼                │
│   ┌─────────┐    ┌──────────┐    ┌──────────┐          │
│   │ Layers  │    │ Autograd │    │Optimizer │          │
│   │Linear/  │    │ backward │    │ SGD/Adam │          │
│   │Attention│    └──────────┘    └──────────┘          │
│   └─────────┘                                           │
│        │                                                │
│        ▼                                                │
│   ┌─────────────────────────────────┐                  │
│   │         Attention               │                  │
│   │  ┌──────┐ ┌──────┐ ┌──────┐    │                  │
│   │  │Head 1│ │Head 2│ │Head N│    │                  │
│   │  └──────┘ └──────┘ └──────┘    │                  │
│   │         ↓ Concat ↓              │                  │
│   │       [Output Projection]       │                  │
│   └─────────────────────────────────┘                  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Features

| Component | Description |
|-----------|-------------|
| **Tensor** | N-dimensional arrays with matmul, add, transpose, concat |
| **Graph** | Computation graph with optional thread-safe arena |
| **Autograd** | Reverse-mode automatic differentiation |
| **Optimizers** | SGD, Adam with momentum, bias correction & gradient clipping |
| **Layers** | Linear (fully connected), Encoder Block |
| **Attention** | Single-head & Multi-head attention (parallel when threading enabled) |
| **Activations** | ReLU, Tanh, Sigmoid, Softmax (row-wise), LayerNorm |
| **Loss** | MSE, Cross-Entropy |
| **Init** | Xavier, Kaiming |
| **Positional** | Sinusoidal positional encoding |
| **Threading** | Optional `std::thread` parallelism via CMake flag |

## Quick Start

```cpp
#include "graph.hpp"
#include "attention.hpp"
#include "optimizer.hpp"
#include "loss.hpp"

int main() {
    Graph g;
    
    // Create multi-head attention (d_model=64, num_heads=4)
    MultiHeadAttention mha(g, 64, 4);
    Adam optimizer(0.001);
    
    // Input: (seq_len=4, d_model=64)
    Tensor input(4, 64, 0.1);
    
    std::size_t num_params = g.arena.size();
    
    for (int epoch = 0; epoch < 100; ++epoch) {
        g.arena.resize(num_params);  // Clear intermediates
        
        NodeID x = g.value(input);
        NodeID output = mha.forward(g, x);
        NodeID loss = Loss::mse(g, output, target);
        
        g.zero_grads();
        g.backward(loss);
        optimizer.step(g);
    }
}
```

## Build

```bash
git clone https://github.com/angelorosu/TransformerCPP.git
cd TransformerCPP
mkdir build && cd build
cmake ..
make
```

### Enable Threading (Optional)

For parallel multi-head attention execution:

```bash
cmake -DENABLE_THREADING=ON ..
make
```

This enables `std::thread`-based parallelism with mutex-protected graph operations.

## Run

```bash
./transformer_main    # Encoder tests & validation
./xor_example         # Classic XOR problem
```

## Test Output

```
=== TEST: Linear Layer (y = 2x + 1) ===
Epoch 0 | Loss: 38.6715
Epoch 400 | Loss: 0.00138646
✓ Linear layer test complete!

=== TEST: Single Attention Head ===
Input shape: (4, 8)
Output shape: (4, 8)
✓ Attention head test complete!

=== TEST: Multi-Head Attention ===
Input shape: (4, 64)
Output shape: (4, 64)
Num heads: 4
✓ Multi-head attention test complete!

=== TEST: Attention Actually Learns ===
Epoch 0 | Loss: 0.00915111
Epoch 40 | Loss: 0.00146473
✓ Attention is learning!

=== TEST: Encoder Block ===
Input shape: (4, 32)
Output shape: (4, 32)
✓ Encoder block test complete!

=== TEST: Encoder Learns to Predict Next Number ===
Task: [1,2,3,4] → [2,3,4,5] (predict next number)
Epoch 0 | Loss: 0.168993
Epoch 150 | Loss: 0.0479694
✓ Encoder is learning!
```

## How Attention Works

```
1. Project input X into Q, K, V:
   Q = X @ W_q
   K = X @ W_k  
   V = X @ W_v

2. Scaled dot-product attention:
   Scores = Q @ K^T / √d_k
   Weights = softmax(Scores)  ← row-wise!
   Output = Weights @ V

3. Multi-head: Run N heads in parallel, concat, project
   MultiHead(X) = Concat(head_1, ..., head_n) @ W_o
```

## Project Structure

```
TransformerCPP/
├── include/
│   ├── tensor.hpp          # Tensor class
│   ├── node.hpp            # Node & Op enum
│   ├── graph.hpp           # Computation graph (thread-safe optional)
│   ├── optimizer.hpp       # SGD, Adam (with gradient clipping)
│   ├── layers.hpp          # Linear layer
│   ├── attention.hpp       # AttentionHead, MultiHeadAttention
│   ├── transformer_block.hpp   # EncoderBlock (Pre-LN)
│   ├── positional_encoding.hpp # Sinusoidal PE
│   ├── threading.hpp       # parallel_for helper
│   ├── loss.hpp            # Loss functions
│   └── init.hpp            # Weight initialization
├── src/
│   ├── tensor.cpp
│   ├── graph.cpp
│   ├── optimizer.cpp
│   ├── layers.cpp
│   ├── attention.cpp
│   ├── transformer_block.cpp
│   ├── init.cpp
│   └── main.cpp
├── examples/
│   └── xor.cpp
├── CMakeLists.txt
└── README.md
```

## Supported Operations

| Category | Operations |
|----------|------------|
| **Math** | add, mul, div, matmul, transpose, concat |
| **Activations** | relu, tanh, sigmoid, softmax |
| **Normalization** | layer_norm |
| **Reduction** | mean, log |
| **Attention** | scaled dot-product, multi-head |

## Roadmap

- [x] Core tensor operations
- [x] Reverse-mode autodiff
- [x] SGD & Adam optimizers (with gradient clipping)
- [x] Linear layer
- [x] Activation functions (row-wise softmax)
- [x] Single Attention Head
- [x] Multi-Head Attention
- [x] Concat operation
- [x] Positional Encoding (sinusoidal)
- [x] Transformer Encoder Block (Pre-LN architecture)
- [ ] Stack N encoder blocks
- [ ] Decoder block with causal masking
- [ ] Thread pool for parallel heads
- [ ] Full Transformer encoder/decoder
- [ ] GPU support (CUDA)

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy

## License

MIT License

## Author

**Angelo** - [GitHub](https://github.com/angelorosu)