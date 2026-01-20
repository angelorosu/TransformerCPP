#include <iostream>
#include <tensor.hpp>
#include <graph.hpp>
#include <optimizer.hpp>
#include <layers.hpp>
#include <loss.hpp>

int main()
{
    std::cout <<" XOR Problem (Non Linear) \n" ;

    Graph g;
    Linear layer1(g, 2 ,8); // 2 inputs -> 8 hidden
    Linear layer2(g, 8, 1 ); // 8 hidden layers -> 1 output
    Adam optimizer(0.1);

    double X[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double Y[4] = {0,1,1,0};

    for (int epoch =0; epoch < 1000; ++epoch)
    {
        double total_loss = 0.0;

        for (int i =0; i< 4 ; ++i)
        {
            Tensor x_in(1,2,0.0);
            x_in.data[0] = X[i][0];
            x_in.data[1] = X[i][1];

            NodeID x = g.value(x_in);
            NodeID target = g.value(Tensor(1,1,Y[i]));

            //forward = sigmoid(layer2(tanh(layer1)))
            NodeID h = g.tanh(layer1.forward(g,x));
            NodeID pred = g.sigmoid(layer2.forward(g,h));
            NodeID loss = Loss::mse(g, pred, target);

            total_loss += g.arena[loss].data.data[0];

            g.zero_grads();
            g.backward(loss);
            optimizer.step(g);

            
        }
        if (epoch % 100 == 0)
            {
                std::cout << "Epoch : " << epoch << "Loss : " << total_loss/4 << "\n";
            }
    }

    
    std::cout << "\nXOR Results:\n";
    for (int i = 0; i < 4; ++i) {
        Tensor x_in(1, 2, 0.0);
        x_in.data[0] = X[i][0];
        x_in.data[1] = X[i][1];
        
        NodeID x = g.value(x_in);
        NodeID h = g.tanh(layer1.forward(g, x));
        NodeID pred = g.sigmoid(layer2.forward(g, h));
        
        double result = g.arena[pred].data.data[0];
        std::cout << (int)X[i][0] << " XOR " << (int)X[i][1] 
                  << " = " << (result > 0.5 ? 1 : 0)
                  << " (raw: " << result << ")\n";
    }
    
    return 0;
}