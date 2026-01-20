#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "tensor.hpp"

using NodeID = std::size_t;

enum class Op : std::uint8_t {
    None,
    Add,
    MatMul,
    Relu, 
    Tanh, 
    Sigmoid, 
    Softmax, 
    LayerNorm, 
    Mul, 
    Div, 
    Sqrt,
    Mean,
    Log,
    Transpose,
    Concat,};

struct Node
{
    Tensor data;
    Tensor grad;

    std::vector<NodeID> prev;
    std::uint8_t prev_count = 0;

    Op op = Op::None;
    int param = 0;
    bool is_parameter = false;
};
