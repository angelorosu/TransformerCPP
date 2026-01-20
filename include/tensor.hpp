#pragma once

#include <cstddef>
#include <vector>

struct Tensor
{
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<double> data;

    Tensor() = default; //create empty tensor with no args
    Tensor(std::size_t r, std::size_t c, double fill);

    std::size_t size() const;
    double get(std::size_t r, std::size_t c) const;
    void set(std::size_t r, std::size_t c, double value);
    bool same_shape(const Tensor& other) const;

    static Tensor zeros_like(const Tensor& x);
    static Tensor ones_like(const Tensor& x);
    static Tensor add(const Tensor& a, const Tensor& b);
    void add_inplace(const Tensor& other);
    void mul_inplace(const Tensor& other);
    static Tensor elemwise_mul(const Tensor& a, const Tensor& b);
    static Tensor matmul(const Tensor& a, const Tensor& b);
    static Tensor transpose(const Tensor& x);

    static Tensor relu(const Tensor& x);
    static Tensor tanh(const Tensor& x);
    static Tensor sigmoid(const Tensor& x);

    static Tensor softmax(const Tensor& x);
    static Tensor layer_norm(const Tensor& x);
    static Tensor elemwise_div(const Tensor& a, const Tensor& b);
    static Tensor elemwise_sqrt(const Tensor& x);
    static Tensor elemwise_sqr(const Tensor& x);
    static double mean(const Tensor& x);
    static Tensor elemwise_log(const Tensor& x);

};
