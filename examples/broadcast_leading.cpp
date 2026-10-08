#include "../include/tensor.h"
#include "../include/tensor_ops.h"
#include <iostream>

int main() {

   Tensor a({3, 1});

    a({0,0}) = 10;
    a({1,0}) = 20;
    a({2,0}) = 30;

    Tensor b = a.broadcastTo({3, 4});

    std::cout << b({0,0}) << "\n";
    std::cout << b({0,3}) << "\n";
    std::cout << b({1,2}) << "\n";
    std::cout << b({2,3}) << "\n";
    

    return 0;
}