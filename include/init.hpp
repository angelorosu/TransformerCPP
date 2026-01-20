#pragma once
#include "tensor.hpp"

namespace init
{
    void xavier_uniform(Tensor&t, int in_f, int out_f);

    void kaiming_normal(Tensor& t, int in_f);
}