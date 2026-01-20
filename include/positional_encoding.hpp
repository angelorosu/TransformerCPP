#pragma once
#include "tensor.hpp"
#include <cmath>


namespace PE
{

inline Tensor sinusoidal(std::size_t seq_len, std::size_t d_model)
{
    Tensor encoding(seq_len, d_model, 0.0);

    for(std::size_t pos =0; pos < seq_len; ++pos) //loop through position 
    {
        for (std::size_t i=0; i< d_model; i+=2) //loop through i (feature of model)
        {
            // formula from Att all you need 2017
        
            double freq = std::pow(10000.0, static_cast<double>(i)/ d_model);

            // even index = sin
            encoding.set(pos, i , std::sin(pos/freq));

            // odd index : cos
            if (i+1 < d_model)
            {
                encoding.set(pos, i+ 1, std::cos(pos/freq));
            }
        }
    }
    return encoding;

}



    
} // namespace PE

