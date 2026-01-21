

#include <array>
#include <cstddef>
#include <cstdint>
#include <cassert>
#include "graph.hpp"
#include "node.hpp"
#include "tensor.hpp"


NodeID Graph::value(const Tensor& t)
{
#ifdef ENABLE_THREADING //only used if needed for threading if not ignore 
    std::lock_guard<std::mutex> lock(arena_mutex);
#endif
    Node n;
    n.data = t;
    n.grad = Tensor::zeros_like(t);    
    arena.push_back(n);
    return arena.size() -1;
}

NodeID Graph::parameter(const Tensor& t)
{
    #ifdef ENABLE_THREADING
    std::lock_guard<std::mutex> lock(arena_mutex);
#endif
    Node n;
    n.data = t;
    n.grad = Tensor::zeros_like(t);
    n.is_parameter = true;
    n.op = Op::None;
    n.prev_count =0;

    arena.push_back(n);
    return arena.size() -1;
}

NodeID Graph::add(NodeID a_id, NodeID b_id)
{
    //const Tensor& a = arena[a_id].data; 
    //const Tensor& b = arena[b_id].data; this was the way i done it before but it can cause referencing invalidation in case whole arena moves to new loc to make it bigger
    #ifdef ENABLE_THREADING
    std::lock_guard<std::mutex> lock(arena_mutex);
#endif

    Tensor result = Tensor::add(arena[a_id].data, arena[b_id].data);

    Node out;
    out.data = result;
    out.grad = Tensor::zeros_like(out.data);

    out.prev = {a_id, b_id};
    out.prev_count = 2;
    out.op = Op::Add;

    arena.push_back(out);

    return arena.size() - 1;
}

NodeID Graph::matmul(NodeID a_id, NodeID b_id)
{
    //const Tensor& a = arena[a_id].data;
    //const Tensor& b = arena[b_id].data;
    #ifdef ENABLE_THREADING
    std::lock_guard<std::mutex> lock(arena_mutex);
#endif

    Tensor result = Tensor::matmul(arena[a_id].data, arena[b_id].data);

    Node out;
    out.data = result;
    out.grad = Tensor::zeros_like(out.data);

    out.prev = {a_id,b_id};
    out.prev_count = 2;
    out.op = Op::MatMul;

    arena.push_back(out);

    return arena.size() -1;


}

NodeID Graph::mul(NodeID a_id, NodeID b_id)
{
//    const Tensor& a = arena[a_id].data;
//    const Tensor& b = arena[b_id].data;
    #ifdef ENABLE_THREADING
    std::lock_guard<std::mutex> lock(arena_mutex);
#endif

    Tensor result = Tensor::elemwise_mul(arena[a_id].data, arena[b_id].data);
    Node out;
    out.data = result ;
    out.grad = Tensor::zeros_like(out.data);

    out.prev = {a_id,b_id};
    out.prev_count = 2;
    out.op = Op::Mul;

    arena.push_back(out);

    return arena.size() -1;
}

NodeID Graph::div(NodeID a_id, NodeID b_id)
{
    
    #ifdef ENABLE_THREADING
    std::lock_guard<std::mutex> lock(arena_mutex);
#endif  

    const Tensor& a = arena[a_id].data;
    const Tensor& b = arena[b_id].data;
    Node out;
    out.data = Tensor::elemwise_div(a,b);
    out.grad = Tensor::zeros_like(out.data);

    out.prev = {a_id,b_id};
    out.prev_count = 2;
    out.op = Op::Div;

    arena.push_back(out);

    return arena.size() -1; 
}


NodeID Graph::relu(NodeID a_id)

{   
    #ifdef ENABLE_THREADING
    std::lock_guard<std::mutex> lock(arena_mutex);
#endif
    const Tensor& a = arena[a_id].data;
    

    Node out;
    out.data = Tensor::relu(a);
    out.grad = Tensor::zeros_like(out.data);

    out.prev = {a_id, 0};
    out.prev_count =1;
    out.op = Op::Relu;

    arena.push_back(out);

    return arena.size() - 1; 
}

NodeID Graph::tanh(NodeID a_id)
{   
    #ifdef ENABLE_THREADING
    std::lock_guard<std::mutex> lock(arena_mutex);
#endif
    const Tensor& a = arena[a_id].data;
    

    Node out;
    out.data = Tensor::tanh(a);
    out.grad = Tensor::zeros_like(out.data);

    out.prev = {a_id, 0};
    out.prev_count = 1;
    out.op = Op::Tanh;

    arena.push_back(out);
    
    return arena.size() -1;

}


NodeID Graph::sigmoid(NodeID a_id)
{
    #ifdef ENABLE_THREADING
    std::lock_guard<std::mutex> lock(arena_mutex);
#endif
    const Tensor& a = arena[a_id].data;
    


    Node out;
    out.data = Tensor::sigmoid(a);
    out.grad = Tensor::zeros_like(out.data);

    out.prev = {a_id,0};
    out.prev_count =1;
    out.op = Op::Sigmoid;

    arena.push_back(out);

    return arena.size() - 1;
}

NodeID Graph::softmax(NodeID a_id)
{
    #ifdef ENABLE_THREADING
    std::lock_guard<std::mutex> lock(arena_mutex);
#endif
    const Tensor& a = arena[a_id].data;

    Node out;
    out.data = Tensor::softmax(a);
    out.grad = Tensor::zeros_like(out.data);

    out.prev = {a_id,0};
    out.prev_count =1;
    out.op = Op::Softmax;

    arena.push_back(out);

    return arena.size() - 1;
}


NodeID Graph::layer_norm(NodeID a_id)
{
    #ifdef ENABLE_THREADING
    std::lock_guard<std::mutex> lock(arena_mutex);
#endif
    const Tensor& a = arena[a_id].data;

    Node out;
    out.data = Tensor::layer_norm(a);
    out.grad = Tensor::zeros_like(out.data);

    out.prev = {a_id,0};
    out.prev_count =1;
    out.op = Op::LayerNorm;

    arena.push_back(out);

    return arena.size() - 1;
}

NodeID Graph::log(NodeID a_id)
{
    
  #ifdef ENABLE_THREADING
    std::lock_guard<std::mutex> lock(arena_mutex);
#endif
  Tensor result = Tensor::elemwise_log(arena[a_id].data);

  Node out;
  out.data = std::move(result); //move to avoid copying data
  out.grad = Tensor::zeros_like(out.data);
  out.prev = {a_id, 0};
  out.prev_count = 1;
  out.op = Op::Log;

  arena.push_back(out);

  return arena.size() - 1;

}

NodeID Graph::transpose(NodeID a_id)
{
  #ifdef ENABLE_THREADING
    std::lock_guard<std::mutex> lock(arena_mutex);
#endif
  Tensor result = Tensor::transpose(arena[a_id].data);

  Node out;
  out.data = std::move(result); //move to avoid copying data
  out.grad = Tensor::zeros_like(out.data);
  out.prev = {a_id, 0};
  out.prev_count = 1;
  out.op = Op::Transpose;

  arena.push_back(out);

  return arena.size() - 1;

}

NodeID Graph::concat(const std::vector<NodeID>& inputs)
{
    // concating on axis 1 columns 

    // validate all inputs have same rows
    #ifdef ENABLE_THREADING
    std::lock_guard<std::mutex> lock(arena_mutex);
#endif
    std::size_t rows = arena[inputs[0]].data.rows;
    std::size_t total_cols = 0;

    for( NodeID id : inputs)
    {
        assert(arena[id].data.rows == rows);  // same rows
        total_cols += arena[id].data.cols;
    }

    Tensor result(rows, total_cols,0.0);

    std::size_t col_offset = 0;

    for (NodeID id : inputs)
    {
        const Tensor& t = arena[id].data;
        for (std::size_t r =0; r< rows;++r)
        {
            for (std::size_t c =0; c< t.cols;++c)
            {
                result.set(r, col_offset+c , t.get(r,c));
                // Loop is O(n) - as inner loop runs for that tensors column only not ALL
            }
        }
        col_offset += t.cols;
    }

    Node out;
    out.data = std::move(result);
    out.grad = Tensor::zeros_like(out.data);
    out.prev = inputs;
    out.prev_count = inputs.size();
    out.op = Op::Concat;

    arena.push_back(out);

    return arena.size() - 1;


}




void Graph::build_topo(NodeID v, std::vector<NodeID>& topo, std::vector<std::uint8_t>& seen)
{
    if(seen[v]) return;

    seen[v] = 1; //mark seen

    const Node& n = arena[v]; // store reference in n to the Node stored at index v and use const to avoid modifying of n

    for(std::uint8_t i =0; i< n.prev_count; i++)
    {
        build_topo(n.prev[i],topo,seen);
    }

    // now that we ensured all parents have been visited we can add node v
    topo.push_back(v);
}

void Graph::zero_grads()
{
    for(Node& n : arena)
    {
        for (double& x : n.grad.data) x = 0.0;
    }
}

void Graph::backward(NodeID root)
{
    // need to build topo of all nodes needed to actually compute the root
    std::vector<NodeID> topo;
    std::vector<std::uint8_t> seen(arena.size(),0);
    build_topo(root,topo,seen);

    // clear all grads since we're about to start accumulating with += in backprop
    zero_grads();

    arena[root].grad = Tensor::ones_like(arena[root].data);

    // walk nodes in reverse topo order to cover children first

    // topo.begin is reverse iterator pointing to last element
    // we do backwards to match the chain rule

    for (auto it = topo.rbegin(); it != topo.rend(); ++it)
    {
        NodeID v = *it; // node we're processing
        const Node& n = arena[v]; //const reference to the metadata of that node
        const Tensor& out_grad = arena[v].grad;

        switch(n.op)
        {
            case Op::Add:
            {   
                NodeID a = n.prev[0];
                NodeID b = n.prev[1];
                
                arena[a].grad.add_inplace(out_grad);
                arena[b].grad.add_inplace(out_grad);
                break;
            }

            case Op::MatMul:
            {
                NodeID a = n.prev[0];
                NodeID b = n.prev[1];

                const Tensor& A = arena[a].data;
                const Tensor& B = arena[b].data;

                Tensor At = Tensor::transpose(A);
                Tensor Bt = Tensor::transpose(B);

                Tensor dA = Tensor::matmul(out_grad,Bt);
                Tensor dB = Tensor::matmul(At,out_grad);

                arena[a].grad.add_inplace(dA);
                arena[b].grad.add_inplace(dB);

                break;
            }

            case Op::Relu:
            {
                NodeID a = n.prev[0];
                const Tensor& a_data = arena[a].data;   
                Tensor& a_grad = arena[a].grad;

                // chain rule grad_in = grad_out * (1 if x>0 else 0 )
                for (std::size_t i=0; i < out_grad.data.size(); ++i)
                {
                    if (a_data.data[i]>0.0)
                        {
                            a_grad.data[i] += out_grad.data[i];
                        }
                }
                break;
            }

            case Op::Tanh:
            {
                NodeID a = n.prev[0];

                for (std::size_t i =0; i < out_grad.data.size(); ++i)
                {
                    double y = n.data.data[i];

                    // chain rule: grad_in = grad_out * (1-y^2)
                    arena[a].grad.data[i] += out_grad.data[i] * (1.0 - y*y);
                } 

                break;
            }

            case Op::Sigmoid:
            {
                NodeID a = n.prev[0];

                for (std::size_t i =0; i < out_grad.data.size(); ++i)
                {
                    double y = n.data.data[i];

                    // chain rule: grad_in = grad_out * y(1-y)
                    arena[a].grad.data[i] += out_grad.data[i] * y * (1.0 - y);
                } 

                break;
            }
            case Op::Softmax:
            {
                NodeID a = n.prev[0];
                const Tensor& y = n.data; // softmax output
                Tensor& a_grad = arena[a].grad;

                // Row-wise softmax backward
                // grad_in = y * (grad_out - sum(grad_out * y))
                for (std::size_t r = 0; r < y.rows; ++r)
                {
                    // Compute sum for this row: sum(grad_out * y)
                    double sum = 0.0;
                    for (std::size_t c = 0; c < y.cols; ++c)
                    {
                        sum += out_grad.get(r, c) * y.get(r, c);
                    }

                    // Compute gradient for this row
                    for (std::size_t c = 0; c < y.cols; ++c)
                    {
                        double dy = y.get(r, c) * (out_grad.get(r, c) - sum);
                        a_grad.set(r, c, a_grad.get(r, c) + dy);
                    }
                }

                break;
            }
            case Op::Mul:
            {
                NodeID a = n.prev[0];
                NodeID b = n.prev[1];
                
                const Tensor& A = arena[a].data;
                const Tensor& B = arena[b].data;
                
                // d/da (a*b) = b, d/db (a*b) = a
                Tensor dA = Tensor::elemwise_mul(B, out_grad);
                Tensor dB = Tensor::elemwise_mul(A, out_grad);
                
                arena[a].grad.add_inplace(dA);
                arena[b].grad.add_inplace(dB);
                
                break;
            }

            case Op::Div:
            {
                NodeID a = n.prev[0];
                NodeID b = n.prev[1];
                
                const Tensor& A = arena[a].data;
                const Tensor& B = arena[b].data;
                
                // d/da (a/b) = 1/b, d/db (a/b) = -a/b^2
                Tensor dA = Tensor::elemwise_div(out_grad, B);
                
                Tensor b_sq = Tensor::elemwise_sqr(B);  // b squared, not sqrt
                Tensor neg_a = Tensor::elemwise_mul(A, Tensor::ones_like(A));  // negate
                for(auto& x : neg_a.data) x = -x;
                Tensor dB = Tensor::elemwise_div(neg_a, b_sq);
                Tensor dB_scaled = Tensor::elemwise_mul(dB, out_grad);
                
                arena[a].grad.add_inplace(dA);
                arena[b].grad.add_inplace(dB_scaled);
                
                break;
            }

            case Op::LayerNorm:
            {
                NodeID a_id = n.prev[0];
                const Tensor& x = arena[a_id].data;
                //const Tensor& y = n.data; // The normalized output
                Tensor& x_grad = arena[a_id].grad;
                
                constexpr double eps = 1e-12;

                for (std::size_t r = 0; r < x.rows; ++r) {
                    std::size_t N = x.cols;
                    
                    // 1. Re-calculate Mean and StdDev for this row
                    double sum = 0.0;
                    for (std::size_t c = 0; c < N; ++c) sum += x.get(r, c);
                    double mean = sum / N;

                    double var_sum = 0.0;
                    for (std::size_t c = 0; c < N; ++c) {
                        double diff = x.get(r, c) - mean;
                        var_sum += diff * diff;
                    }
                    double std_dev = std::sqrt((var_sum / N) + eps);

                    // 2. Intermediate sums needed for the formula
                    double sum_dy = 0.0;
                    double sum_dy_xhat = 0.0;
                    for (std::size_t c = 0; c < N; ++c) {
                        double dy = out_grad.get(r, c);
                        double x_hat = (x.get(r, c) - mean) / std_dev;
                        sum_dy += dy;
                        sum_dy_xhat += dy * x_hat;
                    }

                    // 3. Calculate final input gradient for each element in the row
                    for (std::size_t c = 0; c < N; ++c) {
                        double dy = out_grad.get(r, c);
                        double x_hat = (x.get(r, c) - mean) / std_dev;
                        
                        double dx = (1.0 / (N * std_dev)) * ( (N * dy) - sum_dy - (x_hat * sum_dy_xhat) );
                        
                        // Accumulate into the parent's grad
                        x_grad.set(r, c, x_grad.get(r, c) + dx);
                    }
                }
                break;
            }
            case Op::Log:
            {
              NodeID a_id = n.prev[0];
              const Tensor& a_data = arena[a_id].data;
              Tensor& a_grad = arena[a_id].grad;

              for (std::size_t i =0; i < n.grad.data.size(); ++i)
              {
                a_grad.data[i] += n.grad.data[i] * (1.0 / (a_data.data[i] + 1e-12));
              }
              break;
            }
            case Op::Transpose:
            {
                NodeID a_id = n.prev[0];

                Tensor d_transposed = Tensor::transpose(out_grad);
                arena[a_id].grad.add_inplace(d_transposed);
                break;
            }

            case Op::None:
            default:
                break;
                    

    }


}


}
