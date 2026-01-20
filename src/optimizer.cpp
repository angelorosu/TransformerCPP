#include "optimizer.hpp"

//constructor logic  - SGD function in sgd class
SGD::SGD(double learning_rate) : lr(learning_rate) {}

void SGD::step(Graph& g)
{
    // loop through actual node(not copy) in arena
    for (Node& n : g.arena)
    {
        if (n.is_parameter)
        {
            // loop through every number in nodes data

            for (std::size_t i=0; i< n.data.data.size(); ++i)
            {
                n.data.data[i] -= lr*n.grad.data[i];
            }
        }
    }

}

Adam::Adam(double learning_rate) 
    : lr(learning_rate), beta1(0.9), beta2(0.999), eps(1e-8), t(0) {}

void Adam::step(Graph& g)
{
    t++;

    const double bias_correction1 = 1.0 - std::pow(beta1, t);
    const double bias_correction2 = 1.0 - std::pow(beta2, t);
    
    // Gradient clipping: compute global norm first
    double global_norm_sq = 0.0;
    for (std::size_t i = 0; i < g.arena.size(); ++i) {
        if (!g.arena[i].is_parameter) continue;
        for (double val : g.arena[i].grad.data) {
            if (std::isfinite(val)) {
                global_norm_sq += val * val;
            }
        }
    }
    double global_norm = std::sqrt(global_norm_sq);
    
    // Clip gradients if norm exceeds max_grad_norm
    double clip_coef = 1.0;
    if (global_norm > max_grad_norm) {
        clip_coef = max_grad_norm / global_norm;
    }
    
    for (std::size_t i = 0; i < g.arena.size(); ++i) {
        if (!g.arena[i].is_parameter) continue;
        
        // Initialize m, v for this node if not exists
        if (m.find(i) == m.end()) {
            m[i] = Tensor::zeros_like(g.arena[i].data);
            v[i] = Tensor::zeros_like(g.arena[i].data);
        }
        
        Tensor& data = g.arena[i].data;
        Tensor& grad = g.arena[i].grad;
        
        for (std::size_t j = 0; j < data.data.size(); ++j) {
            // Skip NaN/inf gradients entirely
            if (!std::isfinite(grad.data[j])) continue;
            
            // Apply clipping to gradient
            double clipped_grad = grad.data[j] * clip_coef;
            
            // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            m[i].data[j] = beta1 * m[i].data[j] + (1.0 - beta1) * clipped_grad;
            
            // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            v[i].data[j] = beta2 * v[i].data[j] + (1.0 - beta2) * clipped_grad * clipped_grad;
            
            // Bias corrected estimates
            double m_hat = m[i].data[j] / bias_correction1;
            double v_hat = v[i].data[j] / bias_correction2;
            
            // Adam update
            double update = lr * m_hat / (std::sqrt(v_hat) + eps);
            
            // Only update if finite (prevent NaN/inf)
            if (std::isfinite(update)) {
                data.data[j] -= update;
            }
        }
    }
}