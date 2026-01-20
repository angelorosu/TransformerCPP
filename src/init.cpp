#include "init.hpp"
#include <cmath>
#include <random>

void init::xavier_uniform(Tensor& t, int in_f, int out_f)
{
    double limit = std::sqrt(6.0/(in_f+out_f));

    // RNG
    std::default_random_engine gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-limit, limit);

    for (double& val : t.data)
    {
        val = dist(gen);
    }
}