#include "tensor.hpp"
#include <cassert>

Tensor::Tensor(std::size_t r, std::size_t c, double fill)
    {
        rows = r;
        cols = c;
        data = std::vector<double>(r*c, fill);
    }  

// Total number of elements
std::size_t Tensor::size() const
{
    return rows*cols;
}

//read element at (r,c)
double Tensor::get(std::size_t r, std::size_t c) const
{
    assert(r < rows && c < cols);
    std::size_t idx = r * cols + c;
    return data[idx];
}

// set element at (r,c)
void Tensor::set(std::size_t r, std::size_t c, double value)
{
    assert(r < rows && c < cols);
    std::size_t idx = r * cols + c;
    data[idx] = value;
}
    

// check shape
bool Tensor::same_shape(const Tensor& other) const
{
    return rows == other.rows && cols == other.cols;
}

// tensor filled with 0s
Tensor Tensor::zeros_like(const Tensor& x)
{
    return Tensor(x.rows, x.cols, 0.0);
}

// tensor filled with 1s
Tensor Tensor::ones_like(const Tensor& x)
{
    return Tensor(x.rows, x.cols, 1.0);
}

// element wise addition
Tensor Tensor::add(const Tensor& a, const Tensor& b)
{
    assert(a.same_shape(b));
    Tensor out(a.rows, a.cols, 0.0);
    for(std::size_t i=0; i < out.data.size(); i++)
    {
        out.data[i] = a.data[i] + b.data[i];
    }
    return out;
}

// inplace add
void Tensor::add_inplace(const Tensor& other)
{
    assert(same_shape(other));
    for (std::size_t i = 0; i < data.size(); i++)
    {
        data[i] += other.data[i];
    }
}

void Tensor::mul_inplace(const Tensor& other)
{
    assert(same_shape(other));
    for (std::size_t i = 0; i< data.size(); i++)
    {
        data[i] *= other.data[i];
    }
}

// element wise multiplication
Tensor Tensor::elemwise_mul(const Tensor& a, const Tensor& b)
{
    assert(a.same_shape(b));
    Tensor out(a.rows, a.cols, 0.0);
    for (std::size_t i=0; i < out.data.size(); i++)
    {
        out.data[i] = a.data[i] * b.data[i];
    }
    return out; 
}

Tensor Tensor::matmul(const Tensor& a, const Tensor& b)
{
    assert(a.cols == b.rows); //ensures valid matrix mul

    Tensor out(a.rows, b.cols, 0.0); //pre-allocate memory for the out-tensor

    for (std::size_t i=0; i < a.rows; ++i) //rows
    {
        for (std::size_t k =0; k < a.cols; ++k ) //shared dimension over k
        {
            const double temp = a.data[i*a.cols + k];

            for (std::size_t j =0; j< b.cols; ++j) //cols
            {
                out.data[i* out.cols + j] += temp * b.data[k*b.cols + j];
            }
        }
    }
    return out;

}

Tensor Tensor::transpose(const Tensor& x)
{
    Tensor out(x.cols, x.rows,0.0);

    for (std::size_t i =0; i < x.rows; ++i)
    {
        for (std::size_t j=0; j<x.cols; ++j)
        {
            out.data[j*out.cols + i] = x.data[i*x.cols +j];
        }
    }

    return out;
}
Tensor Tensor::relu(const Tensor& x)
{
    Tensor out(x.rows, x.cols,0.0);

    for(std::size_t i=0; i < x.data.size(); ++i)
    {
        out.data[i] = (x.data[i] > 0.0) ? x.data[i] : 0.0;
    }

    return out;

}

Tensor Tensor::tanh(const Tensor& x)
{
    Tensor out(x.rows, x.cols,0.0);

    for(std::size_t i=0; i < x.data.size(); ++i)
    {
        out.data[i] = std::tanh(x.data[i]);
    }

    return out;
}

Tensor Tensor::sigmoid(const Tensor& x)
{
    Tensor out(x.rows, x.cols,0.0);

    for(std::size_t i=0; i < x.data.size(); ++i)
    {
        out.data[i] = 1 / (1 + std::exp(-x.data[i]));  
    }

    return out;
}

Tensor Tensor::softmax(const Tensor& x)
{
    Tensor out(x.rows, x.cols, 0.0);

    // Row-wise softmax (each row sums to 1)
    for (std::size_t r = 0; r < x.rows; ++r)
    {
        // Find max in this row for numerical stability
        double max_val = x.get(r, 0);
        for (std::size_t c = 1; c < x.cols; ++c)
        {
            if (x.get(r, c) > max_val) max_val = x.get(r, c);
        }

        // Calculate sum of exponentials for this row
        double sum = 0.0;
        for (std::size_t c = 0; c < x.cols; ++c)
        {
            sum += std::exp(x.get(r, c) - max_val);
        }

        // Normalize this row
        for (std::size_t c = 0; c < x.cols; ++c)
        {
            out.set(r, c, std::exp(x.get(r, c) - max_val) / sum);
        }
    }

    return out;
}
Tensor Tensor::layer_norm(const Tensor& x)
{
    Tensor out(x.rows,x.cols,0.0);
    constexpr double eps = 1e-12;

    for(std::size_t r = 0; r < x.rows; ++r)
    {
        //calc mean for this row
        double sum = 0.0;
        for (std::size_t c =0; c< x.cols; ++c)
        {
            sum+= x.get(r,c);
        }
        double mean = sum / static_cast<double>(x.cols);

        //variance for this row
        double var_sum = 0.0;
        for (std::size_t c =0; c< x.cols; ++c)
        {
            double diff = x.get(r,c) - mean;
            var_sum += diff * diff;
        }
        double variance = var_sum / static_cast<double>(x.cols);
        double std_dev = std::sqrt(variance + eps);

        //normalise this row
        for(std::size_t c = 0; c < x.cols; ++c)
        {
            double val = (x.get(r,c) -mean)/std_dev;
            out.set(r,c,val);
        }
    }
    return out;
}

Tensor Tensor::elemwise_div(const Tensor& a, const Tensor& b)
{
    assert(a.same_shape(b));
    constexpr double eps = 1e-12;

    Tensor out(a.rows,a.cols,0.0);

    for(std::size_t i= 0; i < out.data.size(); ++i)
    {
        double denom = (b.data[i] >= 0) ? (b.data[i] + eps) : (b.data[i] - eps);
        out.data[i] = a.data[i]/denom;
    }
    return out;
}

Tensor Tensor::elemwise_sqrt(const Tensor& x)
{
    Tensor out(x.rows,x.cols, 0.0);

    constexpr double eps = 1e-12; // clamp to epsiloon if negative sqrt

    for (std::size_t i =0; i < x.data.size(); ++i)
    {
        out.data[i] = std::sqrt(std::max(x.data[i], eps));
    }
    return out;
}

Tensor Tensor::elemwise_sqr(const Tensor& x)
{
    Tensor out(x.rows, x.cols, 0.0);
    for (std::size_t i = 0; i < x.data.size(); ++i)
    {
        out.data[i] = x.data[i] * x.data[i];
    }
    return out;
}

double Tensor::mean(const Tensor& x)
{
    double sum = 0.0;

    for(std::size_t i =0; i < x.data.size(); ++i)
        sum += x.data[i];

    return sum / static_cast<double>(x.data.size());
}


Tensor Tensor::elemwise_log(const Tensor& x)
{
  Tensor out(x.rows, x.cols, 0.0);

  for (std::size_t i =0; i < x.data.size(); ++i)
  {
    out.data[i] = std::log(x.data[i] +1e-12);
  }
  return out;
}
